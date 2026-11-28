"""Scoring extensions to the pinned author implementation."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def aggregate(scores, mask=None, k=3, peak_weight=0.5):
    """Aggregate the final position axis, excluding padding."""
    mask = torch.ones_like(scores, dtype=torch.bool) if mask is None else torch.broadcast_to(mask.bool(), scores.shape)
    if (mask.sum(-1) == 0).any():
        raise ValueError('A video has no valid local positions')
    values = scores.masked_fill(~mask, -torch.inf).topk(min(k, scores.shape[-1]), dim=-1).values
    valid = torch.isfinite(values)
    mean = values.masked_fill(~valid, 0).sum(-1) / valid.sum(-1)
    return peak_weight * values[..., 0] + (1 - peak_weight) * mean


class ScoringMixin:
    def configure_experiment(self, variant, cfg, multi=False):
        self.variant, self.multi = variant, multi
        self.pooling = variant in ('pooling', 'combined')
        self.gating = variant in ('gating', 'combined')
        self.fixed_weight = cfg['clip_scale_w']
        if self.gating:
            # Do not perturb the base model/data RNG stream.
            with torch.random.fork_rng(devices=[]):
                self.scale_gate = nn.Sequential(nn.Linear(cfg['hidden_size'], 64), nn.GELU(), nn.Linear(64, 1))
                nn.init.zeros_(self.scale_gate[-1].weight)
                nn.init.constant_(self.scale_gate[-1].bias, math.log(self.fixed_weight / (1 - self.fixed_weight)))

    def get_modularized_queries(self, encoded_query, query_mask):
        q = super().get_modularized_queries(encoded_query, query_mask)
        return q.unsqueeze(0) if q.ndim == 1 else q

    def encode_context(self, clip_video_feat, frame_video_feat, video_mask=None):
        frames, clips = super().encode_context(clip_video_feat, frame_video_feat, video_mask)
        self._frame_mask = F.pad(video_mask, (0, frames.shape[1] - video_mask.shape[1]))
        return frames, clips

    def local_scores(self, q, context, normalize=True, mask=None):
        if normalize:
            q, context = F.normalize(q, dim=-1), F.normalize(context, dim=-1)
        scores = torch.einsum('qd,vld->qvl', q, context)
        if self.pooling:
            return aggregate(scores, None if mask is None else mask.unsqueeze(0))
        return scores.max(-1).values

    def weighted_token_wise_intersection(self, text_token, frame_token, attention_mask, video_mask):
        if not self.pooling:
            return super().weighted_token_wise_intersection(text_token, frame_token, attention_mask, video_mask)
        weights = self.text_weight_fc(text_token).squeeze(2)
        weights = weights.masked_fill(~attention_mask.bool(), -torch.inf).softmax(-1)
        scores = torch.einsum('qtd,vld->qvtl', text_token, frame_token)
        pooled = aggregate(scores, video_mask[None, :, None, :])
        return torch.einsum('qvt,qt->qv', pooled, weights)

    def get_pred_from_raw_query(self, query_feat, query_mask, query_labels=None,
                                video_proposal_feat=None, encoded_frame_feat=None,
                                video_mask=None, return_query_feats=False):
        q = self.encode_query(query_feat, query_mask)
        mask = self._frame_mask if video_mask is None else video_mask
        mask = F.pad(mask, (0, encoded_frame_feat.shape[1] - mask.shape[1]))
        clip = self.local_scores(q, video_proposal_feat)
        if self.multi:
            frame_q = self.encode_frame_query(query_feat, query_mask)
            frame = self.get_frame_scale_scores(frame_q, encoded_frame_feat, query_mask, mask)
        else:
            frame = self.local_scores(q, encoded_frame_feat, mask=mask)
        self.last_weight = torch.sigmoid(self.scale_gate(q)) if self.gating else self.fixed_weight
        self.last_fused = self.last_weight * clip + (1 - self.last_weight) * frame
        if return_query_feats:
            clip_raw = self.local_scores(q, video_proposal_feat, normalize=False)
            if self.multi:
                frame_raw = self.get_unnormalized_frame_scale_scores(frame_q, encoded_frame_feat, query_mask, mask)
            else:
                frame_raw = self.local_scores(q, encoded_frame_feat, normalize=False, mask=mask)
            return clip, clip_raw, frame, frame_raw
        return clip, frame


def extend_model(model, variant, cfg, multi=False):
    model.__class__ = type('ExperimentHolmes', (ScoringMixin, model.__class__), {})
    model.configure_experiment(variant, cfg, multi)
    return model


class ValidationMixin:
    def compute_query2ctx_info(self, model, query_eval_loader, ctx_info):
        from Utils.utils import gpu
        rows, metas = [], []
        for batch in query_eval_loader:
            batch = gpu(batch)
            metas.extend(batch[-1])
            scores = []
            # Full candidate set; chunks only bound token-interaction memory.
            for start in range(0, len(ctx_info['video_metas']), 128):
                sl = slice(start, start + 128)
                model.get_pred_from_raw_query(batch[0], batch[1],
                    video_proposal_feat=ctx_info['video_proposal_feat'][sl],
                    encoded_frame_feat=ctx_info['video_feat'][sl],
                    video_mask=ctx_info['video_mask'][sl])
                scores.append(model.last_fused)
            rows.append(torch.cat(scores, dim=1))
        return torch.cat(rows), metas


def extend_validation(validator):
    validator.__class__ = type('ExperimentValidation', (ValidationMixin, validator.__class__), {})
    return validator

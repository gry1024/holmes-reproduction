import torch
from torch import nn
from adapters import aggregate, ScoringMixin


def test_padding_and_short_sequence():
    x = torch.tensor([[-0.8, -0.2, 0., 0.], [0.95, 0.3, 0.2, 0.]])
    mask = torch.tensor([[1,1,0,0], [1,1,1,0]], dtype=torch.bool)
    result = aggregate(x, mask)
    torch.testing.assert_close(result, torch.tensor([-0.35, (0.95 + 1.45/3)/2]))
    torch.testing.assert_close(aggregate(x, mask, k=1), torch.tensor([-0.2, 0.95]))


def test_pooling_gradients():
    x = torch.tensor([[0.9,0.8,0.7,0.1]], requires_grad=True)
    aggregate(x).sum().backward()
    assert (x.grad[0,:3] > 0).all() and x.grad[0,3] == 0


def test_gate_initialization_gradient_and_state():
    class Toy(ScoringMixin, nn.Module):
        pass
    model = Toy()
    cfg = {'clip_scale_w':0.6, 'hidden_size':384}
    model.configure_experiment('gating', cfg)
    q = torch.randn(4,384)
    w = model.scale_gate(q).sigmoid()
    torch.testing.assert_close(w, torch.full((4,1),0.6))
    clip, frame = torch.randn(4,3), torch.randn(4,3)
    loss = torch.nn.functional.cross_entropy((w*clip+(1-w)*frame)/0.1, torch.tensor([0,1,2,0]))
    loss.backward()
    assert model.scale_gate[-1].weight.grad.abs().sum() > 0
    other = Toy()
    other.configure_experiment('gating', cfg)
    other.load_state_dict(model.state_dict())
    torch.testing.assert_close(other.scale_gate(q), model.scale_gate(q))

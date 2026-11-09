import pytest
import torch

from mulg.model import DirectedAttention, ModelConfig, MulG, PaperGRU
from mulg.train import MODALITIES

torch.set_num_threads(1)


def test_bilinear_attention_equations_and_key_mask():
    attention = DirectedAttention(2, 1, 0)
    with torch.no_grad():
        attention.weight.copy_(torch.eye(2)[None])
        attention.out.weight.copy_(torch.eye(2))
        attention.out.bias.zero_()
    q = torch.tensor([[[1.0, 2.0], [9.0, 9.0]]])
    source = torch.tensor([[[2.0, 0.0], [0.0, 3.0], [999.0, 999.0]]])
    actual = attention(q, source, torch.tensor([[True, False]]), torch.tensor([[True, True, False]]))
    expected = (q[:, :1] @ source[:, :2].transpose(1, 2)).softmax(-1) @ source[:, :2]
    torch.testing.assert_close(actual[:, :1], expected)
    assert torch.count_nonzero(actual[:, 1]) == 0


def test_paper_gru_candidate_reset_before_projection_and_padding():
    gru = PaperGRU(1, 1, 1, 0)
    with torch.no_grad():
        gru.x_proj[0].weight.copy_(torch.tensor([[0.3], [0.7], [0.9]]))
        gru.h_gates[0].weight.copy_(torch.tensor([[0.4], [0.6]]))
        gru.h_candidate[0].weight.fill_(0.8)
    state = torch.zeros(1, 1)
    for x in [0.2, 0.5]:
        r = torch.sigmoid(0.3 * x + 0.4 * state)
        z = torch.sigmoid(0.7 * x + 0.6 * state)
        candidate = torch.tanh(0.9 * x + 0.8 * r * state)
        state = (1 - z) * state + z * candidate
    result = gru(torch.tensor([[[0.2], [0.5], [200.0]]]), torch.tensor([[True, True, False]]))
    torch.testing.assert_close(result, state)


@pytest.mark.parametrize("modalities", MODALITIES)
@pytest.mark.parametrize("variant", ["paper", "pytorch"])
def test_all_subsets_gradients_and_padding_invariance(modalities, variant):
    torch.manual_seed(4)
    cfg = ModelConfig(hidden=8, heads=2, attention_layers=1, gru_layers=2, gru_variant=variant)
    model = MulG({"t": 3, "a": 4, "v": 5}, modalities, cfg)
    features, masks = {}, {}
    for m, d, length in [("t", 3, 3), ("a", 4, 5), ("v", 5, 4)]:
        features[m] = torch.randn(2, length, d)
        masks[m] = torch.arange(length)[None, :] < torch.tensor([length, length - 1])[:, None]
    logits = model(features, masks)
    assert logits.shape == (2, 7)
    logits.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert len(model.cross) == len(modalities) * (len(modalities) - 1)
    model.eval()
    with torch.no_grad():
        original = model(features, masks)
        padded = {m: torch.nn.functional.pad(x, (0, 0, 0, 3), value=999) for m, x in features.items()}
        padded_masks = {m: torch.nn.functional.pad(mask, (0, 3), value=False) for m, mask in masks.items()}
        changed = model(padded, padded_masks)
    torch.testing.assert_close(original, changed, atol=2e-6, rtol=2e-5)

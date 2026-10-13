"""Shape / checkpoint tests for the dual-branch model with stand-in encoders (no weights needed)."""
import torch
import torch.nn as nn
from src.models.dual_branch import DualBranchTL, resample_time


class FakeEncoder(nn.Module):
    def __init__(self, out_dim, frames):
        super().__init__()
        self.out_dim, self.frames = out_dim, frames
        self.model = nn.Linear(1, 1)          # stands in for the frozen pretrained net

    def forward(self, wav):
        return torch.randn(wav.shape[0], self.frames, self.out_dim)


def _model():
    return DualBranchTL(FakeEncoder(1024, 199), FakeEncoder(768, 24))


def test_output_shapes():
    ls, le = _model()(torch.randn(3, 64000))
    assert ls.shape == (3, 200) and le.shape == (3, 200)


def test_resample_time():
    assert resample_time(torch.randn(2, 24, 8), 200).shape == (2, 200, 8)


def test_param_budget_under_10M():
    m = _model()
    n = sum(p.numel() for k, p in m.named_parameters() if ".model." not in k)
    assert n < 10e6


def test_trainable_checkpoint_roundtrip():
    a, b = _model(), _model()
    sd = a.trainable_state_dict()
    assert not any(".model." in k for k in sd)
    b.load_trainable_state_dict(sd)
    assert torch.equal(a.sp_head.weight, b.sp_head.weight)


def test_regions_to_mask_formats():
    from src.data.dataset import regions_to_mask
    for cell in ["[(0.4, 1.2)]", "0.40-1.20", "[[20, 60]]", "[(400, 1200)]"]:
        m = regions_to_mask(cell)
        assert m.sum() == 40 and m[20] == 1 and m[59] == 1 and m[60] == 0
    assert regions_to_mask("").sum() == 0 and regions_to_mask(float("nan")).sum() == 0


def test_ablation_modes():
    full = DualBranchTL(FakeEncoder(1024, 199), FakeEncoder(768, 24))
    n_full = sum(p.numel() for k, p in full.named_parameters() if ".model." not in k)
    for mode in ("no_xattn", "speech_only", "env_only"):
        m = DualBranchTL(FakeEncoder(1024, 199), FakeEncoder(768, 24), mode=mode)
        ls, le = m(torch.randn(2, 64000))
        assert ls.shape == (2, 200) and le.shape == (2, 200)
        n = sum(p.numel() for k, p in m.named_parameters() if ".model." not in k)
        assert n < n_full
    assert DualBranchTL(FakeEncoder(1024, 199), None, mode="speech_only").env_encoder is None


def test_full_mode_keys_unchanged():
    keys = set(DualBranchTL(FakeEncoder(1024, 199), FakeEncoder(768, 24)).trainable_state_dict())
    assert {"sp_xattn.in_proj_weight", "en_norm.weight", "sp.gru.weight_ih_l0", "en_head.weight"} <= keys

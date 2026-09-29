"""
src/models/encoders.py  --  frozen pretrained encoders.

XLSREncoder  : wav2vec2 XLS-R 300M (HF format dir: config.json + pytorch_model.bin).
               Returns a learnable softmax-weighted sum of ALL hidden layers, (B, ~199, 1024)
               at 50 Hz. Only the layer weights train; the transformer is frozen.
BEATsEncoder : Microsoft BEATs (iter3+ AS2M checkpoint + the official BEATs.py code).
               Returns last-layer patch features averaged over the 8 frequency patches,
               (B, ~24, 768) at ~6 Hz (one step = 16 fbank frames = 160 ms).

Both encoders are kept in eval() mode permanently (no dropout / no SpecAugment masking)
and run under torch.no_grad().
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn


class _Frozen(nn.Module):
    """Keeps the wrapped pretrained model in eval mode even when .train() is called."""

    def _freeze(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self


class XLSREncoder(_Frozen):
    def __init__(self, model_dir: str, use_bf16: bool = True):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.model = Wav2Vec2Model.from_pretrained(model_dir)
        self._freeze(self.model)
        self.out_dim = self.model.config.hidden_size
        self.n_layers = self.model.config.num_hidden_layers + 1   # + CNN/feature projection output
        self.layer_logits = nn.Parameter(torch.zeros(self.n_layers))
        self.use_bf16 = use_bf16

    def forward(self, wav: torch.Tensor) -> torch.Tensor:          # wav (B, N) float32
        with torch.no_grad():
            x = (wav - wav.mean(dim=1, keepdim=True)) / (wav.std(dim=1, keepdim=True) + 1e-7)
            with torch.autocast(device_type=wav.device.type, dtype=torch.bfloat16,
                                enabled=self.use_bf16 and wav.is_cuda):
                hs = self.model(x, output_hidden_states=True).hidden_states
            hs = torch.stack([h.float() for h in hs], dim=0)         # (L, B, T, D)
        w = torch.softmax(self.layer_logits, dim=0)
        return (w[:, None, None, None] * hs).sum(dim=0)            # (B, T, D)


class BEATsEncoder(_Frozen):
    N_FREQ_PATCHES = 8     # 128 mel bins / 16

    def __init__(self, ckpt_path: str, code_dir: str):
        super().__init__()
        code_dir = str(Path(code_dir).resolve())
        if code_dir not in sys.path:
            sys.path.insert(0, code_dir)        # BEATs.py uses flat imports (from backbone import ...)
        from BEATs import BEATs, BEATsConfig
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ck["cfg"])
        self.model = BEATs(cfg)
        missing, unexpected = self.model.load_state_dict(ck["model"], strict=False)
        bad = [k for k in missing if not k.startswith("predictor")]
        if bad or unexpected:
            raise RuntimeError(f"BEATs checkpoint mismatch: missing={bad[:5]} unexpected={unexpected[:5]}")
        self._freeze(self.model)
        self.out_dim = cfg.encoder_embed_dim

    def forward(self, wav: torch.Tensor) -> torch.Tensor:          # wav (B, N) float32
        with torch.no_grad():
            feats, _ = self.model.extract_features(wav)             # (B, T'*8, D), freq fastest
            B, N, D = feats.shape
            assert N % self.N_FREQ_PATCHES == 0, N
            feats = feats.view(B, N // self.N_FREQ_PATCHES, self.N_FREQ_PATCHES, D).mean(dim=2)
        return feats.float()                                        # (B, T', D)

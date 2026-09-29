"""
src/models/dual_branch.py  --  CompSpoof-TL dual-branch frame-level localiser.

  wav ─┬─ XLS-R (frozen, layer-weighted) ─ resample→200 ─ proj ─ BiGRU ─┐
       │                                                               ├─ cross-attn (both ways, residual)
       └─ BEATs (frozen, last layer)      ─ resample→200 ─ proj ─ BiGRU ─┘
                                                   speech head → (B,200) logits
                                                   env head    → (B,200) logits
One output per 20 ms frame per component; sigmoid(logit) = P(frame is spoofed).
The trainable head is ~6.5M parameters (plus 25 XLS-R layer weights).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def resample_time(x: torch.Tensor, n: int) -> torch.Tensor:
    """(B, T, D) -> (B, n, D) by linear interpolation along time."""
    if x.shape[1] == n:
        return x
    return F.interpolate(x.transpose(1, 2), size=n, mode="linear", align_corners=False).transpose(1, 2)


class Branch(nn.Module):
    def __init__(self, in_dim, d_model, gru_hidden, gru_layers, dropout):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model),
                                  nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(d_model, gru_hidden, num_layers=gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0)

    def forward(self, x):
        return self.gru(self.proj(x))[0]            # (B, T, 2*gru_hidden)


class DualBranchTL(nn.Module):
    def __init__(self, speech_encoder: nn.Module, env_encoder: nn.Module, n_frames: int = 200,
                 d_model: int = 256, gru_hidden: int = 256, gru_layers: int = 2,
                 heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.speech_encoder = speech_encoder
        self.env_encoder = env_encoder
        self.n_frames = n_frames
        h = 2 * gru_hidden
        self.sp = Branch(speech_encoder.out_dim, d_model, gru_hidden, gru_layers, dropout)
        self.en = Branch(env_encoder.out_dim, d_model, gru_hidden, gru_layers, dropout)
        self.sp_xattn = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.en_xattn = nn.MultiheadAttention(h, heads, dropout=dropout, batch_first=True)
        self.sp_norm = nn.LayerNorm(h)
        self.en_norm = nn.LayerNorm(h)
        self.sp_head = nn.Linear(h, 1)
        self.en_head = nn.Linear(h, 1)

    def forward(self, wav: torch.Tensor):
        sp = resample_time(self.speech_encoder(wav), self.n_frames)
        en = resample_time(self.env_encoder(wav), self.n_frames)
        sp, en = self.sp(sp), self.en(en)
        sp2 = self.sp_norm(sp + self.sp_xattn(sp, en, en, need_weights=False)[0])
        en2 = self.en_norm(en + self.en_xattn(en, sp, sp, need_weights=False)[0])
        return self.sp_head(sp2).squeeze(-1), self.en_head(en2).squeeze(-1)

    # --- checkpoints hold only the trainable part (encoders are loaded from disk) ---
    def trainable_state_dict(self):
        return {k: v for k, v in self.state_dict().items()
                if not (k.startswith("speech_encoder.model.") or k.startswith("env_encoder.model."))}

    def load_trainable_state_dict(self, sd):
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [k for k in missing if not (k.startswith("speech_encoder.model.")
                                              or k.startswith("env_encoder.model."))]
        if missing or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")

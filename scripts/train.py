"""
scripts/train.py  --  train / evaluate the CompSpoof-TL dual-branch model.

Smoke test (interactive GPU node, ~5 min):
    PYTHONPATH=. python scripts/train.py --limit_train 512 --limit_val 256 --epochs 1 \
        --out_dir /flash/ee/btech/ee1230446/runs/smoke --no_strict_seed
Full run (PBS):   see scripts/train_pragya.pbs
Evaluate only:    add --eval_only (loads <out_dir>/best.pt)

Outputs in --out_dir:
    last.pt / best.pt        trainable weights + optimizer (resume is automatic from last.pt)
    history.jsonl            one line per epoch (train loss, val frame-EER per component)
    results_<split>.json     full metric suite (all / seen / unseen generators)
    scores_<split>.npz       per-clip frame scores, for later analysis
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import CompSpoofTLDataset                 # noqa: E402
from src.metrics.evaluate import evaluate_suite                 # noqa: E402
from src.metrics.frame_metrics import eer                       # noqa: E402
from src.models.dual_branch import DualBranchTL                 # noqa: E402
from src.models.encoders import BEATsEncoder, XLSREncoder       # noqa: E402
from src.seeds import set_seed                                  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


def get_args():
    a = argparse.ArgumentParser()
    a.add_argument("--data_root", default="/flash/ee/btech/ee3231024/CompSpoof-TL")
    a.add_argument("--xlsr_dir", default="/flash/ee/btech/ee1230446/models/xlsr-300m")
    a.add_argument("--beats_ckpt", default="/flash/ee/btech/ee1230446/models/beats/BEATs_iter3_plus_AS2M.pt")
    a.add_argument("--beats_code", default=str(REPO / "src/models/beats"))
    a.add_argument("--out_dir", required=True)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--no_strict_seed", action="store_true", help="allow seeds outside 13/42/2027")
    a.add_argument("--epochs", type=int, default=30)
    a.add_argument("--batch_size", type=int, default=16)
    a.add_argument("--lr", type=float, default=1e-3)
    a.add_argument("--weight_decay", type=float, default=1e-4)
    a.add_argument("--patience", type=int, default=5, help="early stop after N epochs w/o val improvement")
    a.add_argument("--num_workers", type=int, default=6)
    a.add_argument("--limit_train", type=int, default=None)
    a.add_argument("--limit_val", type=int, default=None)
    a.add_argument("--limit_eval", type=int, default=None)
    a.add_argument("--train_subset", default="all", choices=["all", "c1"],
                   help="c1 = train/select only on C1 all-bonafide splices (C4 shortcut probe)")
    a.add_argument("--label_mode", default="mask", choices=["mask", "regions"],
                   help="regions = splice regions from the manifest as labels (C4 probe)")
    a.add_argument("--eval_only", action="store_true")
    a.add_argument("--skip_final_eval", action="store_true")
    a.add_argument("--log_every", type=int, default=200)
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return a.parse_args()


def make_loader(ds, args, shuffle):
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=shuffle,
                      num_workers=args.num_workers, pin_memory=args.device.startswith("cuda"),
                      persistent_workers=args.num_workers > 0)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ids, sp_s, en_s, sp_m, en_m = [], [], [], [], []
    for wav, sm, em, cid in loader:
        ls, le = model(wav.to(device, non_blocking=True))
        sp_s += list(torch.sigmoid(ls).cpu().numpy()); en_s += list(torch.sigmoid(le).cpu().numpy())
        sp_m += list(sm.numpy().astype(int));           en_m += list(em.numpy().astype(int))
        ids += list(cid)
    return ids, sp_s, en_s, sp_m, en_m


def quick_val(pred):
    _, sp_s, en_s, sp_m, en_m = pred
    e_sp = eer(np.concatenate(sp_s), np.concatenate(sp_m))[0]
    e_en = eer(np.concatenate(en_s), np.concatenate(en_m))[0]
    return e_sp, e_en


def full_eval(pred, manifest, out_dir, split):
    ids, sp_s, en_s, sp_m, en_m = pred
    np.savez_compressed(out_dir / f"scores_{split}.npz", clip_id=np.array(ids),
                        speech_scores=np.stack(sp_s), env_scores=np.stack(en_s),
                        speech_mask=np.stack(sp_m), env_mask=np.stack(en_m))
    res = {"all": evaluate_suite(sp_s, sp_m, en_s, en_m), "n_clips": len(ids)}
    if "unseen_generator" in manifest.columns:
        unseen_col = manifest.set_index("clip_id")["unseen_generator"]
        flag = np.array([str(unseen_col.get(c, "")).strip().lower() not in ("", "nan", "none", "false", "0")
                         for c in ids])
        for name, sel in [("seen", ~flag), ("unseen", flag)]:
            idx = np.where(sel)[0]
            if len(idx) > 0:
                res[name] = evaluate_suite([sp_s[i] for i in idx], [sp_m[i] for i in idx],
                                           [en_s[i] for i in idx], [en_m[i] for i in idx])
                res[f"n_{name}"] = int(len(idx))
    with open(out_dir / f"results_{split}.json", "w") as f:
        json.dump(res, f, indent=2)
    return res


def main():
    args = get_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed, strict=not args.no_strict_seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    with open(out_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print("loading encoders ...", flush=True)
    model = DualBranchTL(XLSREncoder(args.xlsr_dir), BEATsEncoder(args.beats_ckpt, args.beats_code)).to(device)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable params: {n_train/1e6:.2f} M", flush=True)

    # model selection uses the same kind of data/labels as training (for the C4 probe: held-out
    # bonafide splices with splice labels -- the probe never sees spoof labels)
    val_ds = CompSpoofTLDataset(args.data_root, "val", args.limit_val, args.seed,
                                subset=args.train_subset, label_mode=args.label_mode)
    val_loader = make_loader(val_ds, args, shuffle=False)

    if not args.eval_only:
        train_ds = CompSpoofTLDataset(args.data_root, "train", args.limit_train, args.seed,
                                      subset=args.train_subset, label_mode=args.label_mode)
        pw_sp, pw_en = train_ds.pos_weights(seed=args.seed)
        pw_sp, pw_en = float(np.clip(pw_sp, 1, 20)), float(np.clip(pw_en, 1, 20))
        print(f"train {len(train_ds)} | val {len(val_ds)} | pos_weight speech {pw_sp:.2f} env {pw_en:.2f}", flush=True)
        crit_sp = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw_sp, device=device))
        crit_en = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw_en, device=device))
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        train_loader = make_loader(train_ds, args, shuffle=True)

        start_epoch, best, bad_epochs = 0, float("inf"), 0
        if (out_dir / "last.pt").exists():
            ck = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
            model.load_trainable_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_epoch, best, bad_epochs = ck["epoch"] + 1, ck["best"], ck["bad_epochs"]
            print(f"resumed from epoch {ck['epoch']} (best val {best:.4f})", flush=True)

        for epoch in range(start_epoch, args.epochs):
            model.train(); t0 = time.time(); run = 0.0
            for step, (wav, sm, em, _) in enumerate(train_loader, 1):
                wav, sm, em = wav.to(device, non_blocking=True), sm.to(device), em.to(device)
                ls, le = model(wav)
                loss = crit_sp(ls, sm) + crit_en(le, em)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(params, 5.0)
                opt.step()
                run += loss.item()
                if step % args.log_every == 0:
                    print(f"  ep {epoch} step {step}/{len(train_loader)} loss {run/step:.4f} "
                          f"({(time.time()-t0)/step:.2f}s/step)", flush=True)
            sched.step()
            e_sp, e_en = quick_val(predict(model, val_loader, device))
            score = (e_sp + e_en) / 2
            rec = {"epoch": epoch, "train_loss": run / max(step, 1), "val_eer_speech": e_sp,
                   "val_eer_env": e_en, "val_eer_mean": score, "lr": opt.param_groups[0]["lr"],
                   "minutes": (time.time() - t0) / 60}
            print(json.dumps(rec), flush=True)
            with open(out_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            if score < best:
                best, bad_epochs = score, 0
                torch.save({"model": model.trainable_state_dict(), "epoch": epoch, "val": rec}, out_dir / "best.pt")
            else:
                bad_epochs += 1
            torch.save({"model": model.trainable_state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "epoch": epoch, "best": best,
                        "bad_epochs": bad_epochs}, out_dir / "last.pt")
            if bad_epochs >= args.patience:
                print(f"early stop: no improvement for {args.patience} epochs", flush=True)
                break

    if args.skip_final_eval:
        return
    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_trainable_state_dict(ck["model"])
    print(f"final evaluation with best.pt (epoch {ck['epoch']})", flush=True)
    if args.train_subset != "all" or args.label_mode != "mask":
        e_sp, e_en = quick_val(predict(model, val_loader, device))
        with open(out_dir / "results_probe_task.json", "w") as f:
            json.dump({"what": "probe on its own task (val, same subset/labels as training)",
                       "eer_speech": e_sp, "eer_env": e_en, "n": len(val_ds)}, f, indent=2)
        print(f"[probe task, val {args.train_subset}/{args.label_mode}] EER speech {e_sp:.4f} env {e_en:.4f}",
              flush=True)
    for split, limit in [("val", args.limit_val), ("eval", args.limit_eval)]:
        ds = CompSpoofTLDataset(args.data_root, split, limit, args.seed)
        res = full_eval(predict(model, make_loader(ds, args, shuffle=False), device), ds.df, out_dir, split)
        a = res["all"]
        print(f"[{split}] range-EER speech {a['speech']['range_eer']:.4f} env {a['environment']['range_eer']:.4f} "
              f"| event-F1 speech {a['speech']['event_f1']:.3f} env {a['environment']['event_f1']:.3f} "
              f"| clip-correct {a['joint']['clip_correct']:.3f} | macro-F1 {a['utterance']['macro_f1']:.3f}",
              flush=True)


if __name__ == "__main__":
    main()

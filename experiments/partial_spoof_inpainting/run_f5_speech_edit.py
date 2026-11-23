import os
import sys
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
from importlib.resources import files
from omegaconf import OmegaConf
from hydra.utils import get_class
from cached_path import cached_path

from f5_tts.infer.utils_infer import load_checkpoint, load_vocoder
from f5_tts.model import CFM
from f5_tts.model.utils import convert_char_to_pinyin, get_tokenizer

def run_inpainting(
    audio_path,
    origin_text,
    target_text,
    parts_to_edit,
    fix_duration=None,
    output_wav="f5_inpainted.wav",
    nfe_step=32,
    cfg_strength=2.0,
    seed=42,
    device="cpu"
):
    print(f"\n==========================================")
    print(f"Running F5-TTS Bidirectional Speech Editing")
    print(f"Input: {audio_path}")
    print(f"Origin text: {origin_text}")
    print(f"Target text: {target_text}")
    print(f"Parts to edit: {parts_to_edit}")
    print(f"Fix duration: {fix_duration}")
    print(f"Steps: {nfe_step}, CFG: {cfg_strength}, Seed: {seed}")
    print(f"Device: {device}")
    print(f"==========================================")

    exp_name = "F5TTS_v1_Base"
    ckpt_step = 1250000
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{exp_name}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    dataset_name = model_cfg.datasets.name
    tokenizer = model_cfg.model.tokenizer

    mel_spec_type = model_cfg.model.mel_spec.mel_spec_type
    target_sample_rate = model_cfg.model.mel_spec.target_sample_rate
    n_mel_channels = model_cfg.model.mel_spec.n_mel_channels
    hop_length = model_cfg.model.mel_spec.hop_length
    win_length = model_cfg.model.mel_spec.win_length
    n_fft = model_cfg.model.mel_spec.n_fft

    from huggingface_hub import hf_hub_download
    ckpt_path = str(cached_path(f"hf://SWivid/F5-TTS/{exp_name}/model_{ckpt_step}.safetensors"))
    vocab_path = hf_hub_download(repo_id="SWivid/F5-TTS", filename=f"{exp_name}/vocab.txt")
    vocoder = load_vocoder(vocoder_name=mel_spec_type, is_local=False)

    vocab_char_map, vocab_size = get_tokenizer(vocab_path, "custom")

    model = CFM(
        transformer=model_cls(**model_arc, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=dict(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mel_channels,
            target_sample_rate=target_sample_rate,
            mel_spec_type=mel_spec_type,
        ),
        odeint_kwargs=dict(
            method="euler",
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, dtype=None, use_ema=True)
    model.eval()

    # Load audio using soundfile
    audio_np, sr = sf.read(audio_path)
    if audio_np.ndim > 1:
        audio_np = audio_np.mean(axis=-1)
    audio = torch.from_numpy(audio_np).float().unsqueeze(0)
    
    target_rms = 0.1
    rms = torch.sqrt(torch.mean(torch.square(audio)))
    if rms < target_rms:
        audio = audio * target_rms / rms
    if sr != target_sample_rate:
        resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
        audio = resampler(audio)

    audio = audio.to(device)

    with torch.inference_mode():
        original_mel = model.mel_spec(audio)  # (batch, n_mel, n_frames)
        original_mel = original_mel.permute(0, 2, 1)  # (batch, n_frames, n_mel)

    offset_frame = 0
    mel_cond = torch.zeros(1, 0, n_mel_channels, device=device)
    edit_mask = torch.zeros(1, 0, dtype=torch.bool, device=device)
    fix_dur_list = fix_duration.copy() if fix_duration is not None else None

    for part in parts_to_edit:
        start, end = part
        part_dur_sec = end - start if fix_dur_list is None else fix_dur_list.pop(0)

        start_frame = round(start * target_sample_rate / hop_length)
        end_frame = round(end * target_sample_rate / hop_length)
        part_dur_frames = round(part_dur_sec * target_sample_rate / hop_length)

        keep_frames = start_frame - offset_frame

        mel_cond = torch.cat(
            (
                mel_cond,
                original_mel[:, offset_frame:start_frame, :],
                torch.zeros(1, part_dur_frames, n_mel_channels, device=device),
            ),
            dim=1,
        )
        edit_mask = torch.cat(
            (
                edit_mask,
                torch.ones(1, keep_frames, dtype=torch.bool, device=device),
                torch.zeros(1, part_dur_frames, dtype=torch.bool, device=device),
            ),
            dim=-1,
        )
        offset_frame = end_frame

    # Append remaining
    mel_cond = torch.cat((mel_cond, original_mel[:, offset_frame:, :]), dim=1)
    edit_mask = F.pad(edit_mask, (0, mel_cond.shape[1] - edit_mask.shape[-1]), value=True)

    text_list = [target_text]
    final_text_list = convert_char_to_pinyin(text_list)

    duration = mel_cond.shape[1]

    print(f"Total mel frames: {duration} (~{duration * hop_length / target_sample_rate:.2f}s)")
    print(f"Masked frames to inpaint: {(~edit_mask).sum().item()} frames")

    with torch.inference_mode():
        generated, _ = model.sample(
            cond=mel_cond,
            text=final_text_list,
            duration=duration,
            steps=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=-1.0,
            seed=seed,
            edit_mask=edit_mask,
        )
        generated = generated.to(torch.float32)
        gen_mel_spec = generated.permute(0, 2, 1)
        generated_wave = vocoder.decode(gen_mel_spec).cpu()

        if rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

    os.makedirs(os.path.dirname(os.path.abspath(output_wav)), exist_ok=True)
    sf.write(output_wav, generated_wave.squeeze().numpy(), target_sample_rate)
    print(f"[SUCCESS] Saved inpainted audio: {output_wav}")
    return output_wav

if __name__ == "__main__":
    import soundfile as sf
    import numpy as np

    # Prepare directories
    out_dir = "arijit_partial_spoof/f5_results"
    os.makedirs(out_dir, exist_ok=True)

    full_audio_path = "arijit_partial_spoof/arijit_full_31s_ref.wav"
    data, sr = sf.read(full_audio_path)

    # -------------------------------------------------------------
    # Scenario 1: Phrase Replacement ("Thank you guys.")
    # Extract 5.5s snippet from 22.5s to 28.0s
    # In original: "Thank you guys." is 24.74s to 26.24s
    # Local timestamp: start = 24.74 - 22.5 = 2.24s, end = 26.24 - 22.5 = 3.74s
    # -------------------------------------------------------------
    t_start_1 = 22.5
    t_end_1 = 28.0
    snippet_1 = data[int(t_start_1 * sr):int(t_end_1 * sr)]
    snippet_1_path = os.path.join(out_dir, "ref_context_thankyou.wav")
    sf.write(snippet_1_path, snippet_1, sr)

    orig_text_1 = "I want to be honest, because I want to get fair. Thank you guys. We are going to go."
    
    # 1A: Exact Phrase Inpainting with F5-TTS
    run_inpainting(
        audio_path=snippet_1_path,
        origin_text=orig_text_1,
        target_text=orig_text_1,
        parts_to_edit=[[2.24, 3.74]],
        fix_duration=[1.50],
        output_wav=os.path.join(out_dir, "f5_inpaint_thank_you_guys.wav"),
        nfe_step=32,
        seed=42
    )

    # 1B: Replaced Phrase ("Thank you so much.")
    target_text_1b = "I want to be honest, because I want to get fair. Thank you so much. We are going to go."
    run_inpainting(
        audio_path=snippet_1_path,
        origin_text=orig_text_1,
        target_text=target_text_1b,
        parts_to_edit=[[2.24, 3.74]],
        fix_duration=[1.50],
        output_wav=os.path.join(out_dir, "f5_inpaint_thank_you_so_much.wav"),
        nfe_step=32,
        seed=42
    )

    # -------------------------------------------------------------
    # Scenario 2: Intra-sentence Word Replacement ("change my voice" -> "fake my voice")
    # In original: 10.0s to 16.5s
    # "change my voice" is 11.58s to 12.58s
    # Local timestamp: start = 11.58 - 10.0 = 1.58s, end = 12.58 - 10.0 = 2.58s
    # -------------------------------------------------------------
    t_start_2 = 10.0
    t_end_2 = 16.5
    snippet_2 = data[int(t_start_2 * sr):int(t_end_2 * sr)]
    snippet_2_path = os.path.join(out_dir, "ref_context_changemyvoice.wav")
    sf.write(snippet_2_path, snippet_2, sr)

    orig_text_2 = "So I had to change my voice to kind of cut through the thought, the dialogue."
    target_text_2 = "So I had to fake my voice to kind of cut through the thought, the dialogue."

    run_inpainting(
        audio_path=snippet_2_path,
        origin_text=orig_text_2,
        target_text=target_text_2,
        parts_to_edit=[[1.58, 2.58]],
        fix_duration=[1.00],
        output_wav=os.path.join(out_dir, "f5_inpaint_fake_my_voice.wav"),
        nfe_step=32,
        seed=42
    )

    print("\n[ALL INPAINTING JOBS COMPLETED SUCCESSFULLY!]")

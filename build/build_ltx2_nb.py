# -*- coding: utf-8 -*-
"""Generate ltx2_s2v.ipynb — LTX-2 audio-driven I2V ("custom voice") on Colab."""
import json
import sys

cells = []

# Shared launch cell — identical to build_s2v_nb.py (ported from the Diffusion repo's hardened
# wan22_i2v_comfyui_colab.ipynb). Imported from that generator to keep one copy.
import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location("build_s2v_nb_launch", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "launch_cell.py"))
_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
LAUNCH_CELL = _mod.LAUNCH_CELL


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n")})


# --------------------------------------------------------------------------- #
md(r"""
# 🎬 LTX-2 Custom Voice — audio-driven image-to-video (ComfyUI on Colab)

A second speech-to-video engine, parallel to `wan22_s2v.ipynb`. **LTX-2** generates video and
audio *jointly*; this workflow pins the audio to a voice track you supply and lets the model
generate the video to match. One shot, no 77-frame chunking — the clip is exactly as long as the line.

```
 reference clip ──► Qwen3-TTS voice clone (in-graph) ──► speech.wav
                                                            │  LTX-2 audio VAE
 reference image ──► LTX-2 19B (distilled, fp8) ◄───────── fixed audio latent
                          │
                          ▼   2-pass: 8 steps @ half res → 2× latent upscale → 3-step refine
                 720×1280 @ 24 fps talking clip, audio muxed  ──►  ComfyUI_Wan/output/video/
```

| | This notebook (LTX-2) | `wan22_s2v.ipynb` (Wan 2.2 S2V) |
|---|---|---|
| Model | 19B distilled fp8 + Gemma-3 12B text encoder | 14B fp8 + UMT5 |
| Drive footprint | **~41 GB** (own folder `ComfyUI_LTX`) | ~18 GB |
| Length | one shot, = audio length (≤ ~20 s) | 77-frame chunks chained |
| Speed | 8 + 3 steps | 4 steps (lightning) |
| Tuned for | general audio-video | talking / singing |

**Runtime:** an **80 GB card is the comfortable choice** — Colab's A100-80GB, H100 or G4 (Pro+).
A100-40GB works with ComfyUI's automatic offloading but expect slower runs and lower-resolution
first passes; see Troubleshooting. Nothing here runs on a T4.

**Drive:** ~41 GB free for `ComfyUI_LTX/models`. Voices, generated audio, Qwen checkpoints and the
output folder are **shared with `ComfyUI_Wan`**, so anything you made in `voice_explorer.ipynb` is
available here and finished videos land where the LastFrame watcher looks.
""")

# --------------------------------------------------------------------------- #
md("## Step 1 — Verify GPU")
code(r"""
import subprocess
r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True)
name, mem = (r.stdout.strip().split(',') + ['0'])[:2]
mem_gb = int(mem.strip() or 0) / 1024
print(f'GPU: {name.strip()}  ({mem_gb:.0f} GB)')
if mem_gb >= 70:
    print('✅ 80 GB class — full 720x1280 two-pass is comfortable')
elif mem_gb >= 38:
    print('⚠️  40 GB — works with offloading; if you OOM, see Troubleshooting (lower first-pass res / fewer frames)')
else:
    print('❌ Too small for LTX-2 19B + Gemma 12B. Runtime > Change runtime type > A100 (80GB) / H100')
""")

# --------------------------------------------------------------------------- #
md(r"""
## Step 2 — Mount Google Drive

LTX-2 models live in their own `ComfyUI_LTX` folder. Everything voice-related and the output folder
point at `ComfyUI_Wan` so the three notebooks share one set of voices, audio and videos.
""")
code(r"""
from google.colab import drive
import os

drive.mount('/content/drive')

DRIVE_BASE = '/content/drive/MyDrive/ComfyUI_LTX'      # LTX-2 models + this notebook's inputs
WAN_BASE   = '/content/drive/MyDrive/ComfyUI_Wan'      # shared: voices, input_audio, qwen-tts, output
MODELS_DIR = f'{DRIVE_BASE}/models'

for d in [f'{MODELS_DIR}/diffusion_models', f'{MODELS_DIR}/text_encoders', f'{MODELS_DIR}/vae',
          f'{MODELS_DIR}/latent_upscale_models', f'{MODELS_DIR}/loras',
          f'{DRIVE_BASE}/input_images',
          f'{WAN_BASE}/output', f'{WAN_BASE}/voices', f'{WAN_BASE}/input_audio',
          f'{WAN_BASE}/models/qwen-tts/Qwen']:
    os.makedirs(d, exist_ok=True)

print(f'✅ Drive mounted: {DRIVE_BASE}  (shared voices/audio/output in {WAN_BASE})')
""")

# --------------------------------------------------------------------------- #
md(r"""
## Step 3 — Install ComfyUI + Custom Nodes

LTX-2 nodes are core ComfyUI (needs a recent build — this does `git pull`). The workflow also uses
**KJNodes**, **ComfyUI_essentials**, **comfy_mtb** (audio duration / Whisper), **VideoHelperSuite**
and **ComfyUI-TD-Qwen3TTS** for the in-graph voice clone.
""")
code(r"""
import os, subprocess
os.chdir('/content')

# ComfyUI — clone fresh if missing/corrupt, else update
if os.path.exists('/content/ComfyUI'):
    ok = subprocess.run(['git', 'rev-parse', '--git-dir'], cwd='/content/ComfyUI',
                        capture_output=True).returncode == 0
    if ok:
        !cd /content/ComfyUI && git pull -q
        print('✅ ComfyUI updated')
    else:
        !rm -rf /content/ComfyUI && git clone -q https://github.com/comfyanonymous/ComfyUI.git
        print('✅ ComfyUI re-cloned (was corrupt)')
else:
    !git clone -q https://github.com/comfyanonymous/ComfyUI.git
    print('✅ ComfyUI cloned')

# Core requirements (flag avoids reinstalling every session)
if not os.path.exists('/content/comfyui_reqs_installed'):
    !pip install -q -r /content/ComfyUI/requirements.txt
    open('/content/comfyui_reqs_installed', 'w').close()
    print('✅ Requirements installed')
else:
    print('✅ Requirements already installed')

# Custom nodes
for name, repo in [('ComfyUI-Manager', 'https://github.com/ltdrdata/ComfyUI-Manager.git'),
                   ('ComfyUI-VideoHelperSuite', 'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git'),
                   ('ComfyUI-KJNodes', 'https://github.com/kijai/ComfyUI-KJNodes.git'),
                   ('ComfyUI_essentials', 'https://github.com/cubiq/ComfyUI_essentials.git'),
                   ('comfy_mtb', 'https://github.com/melMass/comfy_mtb.git'),                       # Audio Duration / Whisper nodes
                   ('ComfyUI-TD-Qwen3TTS', 'https://github.com/AICoderTudou/ComfyUI-TD-Qwen3TTS.git')]:  # in-graph voice clone
    path = f'/content/ComfyUI/custom_nodes/{name}'
    if not os.path.exists(path):
        !git clone -q {repo} {path}
        if os.path.exists(f'{path}/requirements.txt'):
            !pip install -q -r {path}/requirements.txt
        print(f'✅ {name} installed')
    else:
        !cd {path} && git pull -q
        print(f'✅ {name} ready')

# Qwen3-TTS is incompatible with transformers 5.x
!pip install -q "transformers>=4.57,<5"

# ffmpeg (for video utilities)
if subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode != 0:
    !apt-get install -y -q ffmpeg
print('\n✅ All installs complete')
""")

# --------------------------------------------------------------------------- #
md("## Step 4 — Link Drive model folders into ComfyUI")
code(r"""
import os
COMFY_MODELS = '/content/ComfyUI/models'

links = {f'{COMFY_MODELS}/{f}': f'{MODELS_DIR}/{f}'
         for f in ['diffusion_models', 'text_encoders', 'vae', 'latent_upscale_models', 'loras']}
links[f'{COMFY_MODELS}/Qwen3-TTS-Models'] = f'{WAN_BASE}/models/qwen-tts/Qwen'   # TD Qwen nodes look here; shared checkpoints
links['/content/ComfyUI/output'] = f'{WAN_BASE}/output'                          # → output/video/ for the LastFrame watcher

for dst, src in links.items():
    if os.path.exists(dst) and not os.path.islink(dst):
        !rm -rf {dst}
    if not os.path.islink(dst):
        os.symlink(src, dst)
    print(f'  ✅ {os.path.basename(dst)} → Drive')
print('\n✅ Folders linked')
""")

# --------------------------------------------------------------------------- #
md(r"""
## Step 5 — Download LTX-2 Models (first run only, ~41 GB, 30–45 min)

All ungated. Filenames match the patched workflow's dropdowns exactly.

| File | Folder | Size |
|---|---|---|
| `ltx-2-19b-distilled-fp8_transformer_only.safetensors` | diffusion_models | 21.5 GB |
| `gemma_3_12B_it_fp8_scaled.safetensors` | text_encoders | 13.2 GB |
| `ltx-2-19b-embeddings_connector_distill_bf16.safetensors` | text_encoders | 2.9 GB |
| `LTX2_video_vae_bf16.safetensors` | vae | 2.4 GB |
| `LTX2_audio_vae_bf16.safetensors` | vae | 0.2 GB |
| `ltx-2-spatial-upscaler-x2-1.0.safetensors` | latent_upscale_models | 1.0 GB |
| Qwen3-TTS `1.7B-Base` + tokenizer (shared) | ComfyUI_Wan/models/qwen-tts | 3.5 GB |
""")
code(r"""
import os
KJ  = 'https://huggingface.co/Kijai/LTXV2_comfy/resolve/main'
CO  = 'https://huggingface.co/Comfy-Org/ltx-2/resolve/main/split_files'
LT  = 'https://huggingface.co/Lightricks/LTX-2/resolve/main'

models = [
    ('LTX-2 19B distilled fp8 (~21.5GB)',
     f'{KJ}/diffusion_models/ltx-2-19b-distilled-fp8_transformer_only.safetensors',
     f'{MODELS_DIR}/diffusion_models/ltx-2-19b-distilled-fp8_transformer_only.safetensors'),
    ('Gemma-3 12B text encoder fp8 (~13.2GB)',
     f'{CO}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors',
     f'{MODELS_DIR}/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors'),
    ('Embeddings connector (distill) (~2.9GB)',
     f'{KJ}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors',
     f'{MODELS_DIR}/text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors'),
    ('LTX-2 video VAE (~2.4GB)',
     f'{KJ}/VAE/LTX2_video_vae_bf16.safetensors',
     f'{MODELS_DIR}/vae/LTX2_video_vae_bf16.safetensors'),
    ('LTX-2 audio VAE (~0.2GB)',
     f'{KJ}/VAE/LTX2_audio_vae_bf16.safetensors',
     f'{MODELS_DIR}/vae/LTX2_audio_vae_bf16.safetensors'),
    ('Spatial upscaler x2 (~1.0GB)',
     f'{LT}/ltx-2-spatial-upscaler-x2-1.0.safetensors',
     f'{MODELS_DIR}/latent_upscale_models/ltx-2-spatial-upscaler-x2-1.0.safetensors'),
]
for label, url, dest in models:
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f'  ✅ Present ({os.path.getsize(dest)/1024**3:.1f}GB): {label}')
    else:
        print(f'  ⬇️  Downloading: {label}')
        !wget -q --show-progress -O "{dest}" "{url}"
        print(f'  ✅ Done ({os.path.getsize(dest)/1024**3:.1f}GB): {label}')

# Qwen3-TTS (shared with the other notebooks)
from huggingface_hub import snapshot_download
def _complete(d):
    # config.json alone means an interrupted download; require the actual weights
    w = f'{d}/model.safetensors'
    return os.path.exists(w) and os.path.getsize(w) > 100 * 1024**2

for name in ['Qwen3-TTS-Tokenizer-12Hz', 'Qwen3-TTS-12Hz-1.7B-Base']:
    dest = f'{WAN_BASE}/models/qwen-tts/Qwen/{name}'
    if _complete(dest):
        print(f'  ✅ Present ({os.path.getsize(f"{dest}/model.safetensors")/1024**3:.1f}GB): {name}')
    else:
        print(f'  ⬇️  Downloading: {name}')
        snapshot_download(repo_id=f'Qwen/{name}', local_dir=dest)   # resumes partial files
        print(f'  ✅ Done: {name}' if _complete(dest) else f'  ⚠️  {name} incomplete — re-run this cell')
print('\n✅ All models ready')
""")

# --------------------------------------------------------------------------- #
md(r"""
## Step 6 — Launch ComfyUI + public URL

**Keep this cell running.** `TUNNEL = "colab"` (default) opens ComfyUI same-origin via Colab's
own port proxy — no 403 host/origin errors, no incognito dance. On a 40 GB card that OOMs, set
`EXTRA_ARGS = ['--lowvram']` before running.

Then **Workflow → Open → `ComfyUI_Wan/ltx2_custom_voice.json`** (Drive) — see the next section.
""")
code(LAUNCH_CELL)

# --------------------------------------------------------------------------- #
md(r"""
---
## 🎛️ The workflow — `ltx2_custom_voice.json`

This is the *"ltx custom voice — AI Verse"* graph, patched so every dropdown matches the files
Step 5 downloads and so videos save to `output/video/`. Seven groups, left to right:

| Group | What to touch |
|---|---|
| **Models** | nothing — `UNETLoader` = distilled fp8, `DualCLIPLoader` = Gemma fp8 + connector (type `ltxv`), video VAE, audio VAE (`VAELoader KJ`), spatial upscaler. `Camera lora` is bypassed; leave it. |
| **Qwen TTS** | `LoadAudio` → your reference clip (5–15 s; Utility A copies it in). `TD Qwen3 TTS Voice Clone` → **type the line** in the first field. The bypassed *Audio To Text (mtb)* Whisper node auto-transcribes the reference — enable it if you don't paste a transcript into `ref_text`. |
| **I2V** | `LoadImage` → reference image. `WIDTH` / `HEIGHT` constants (720 × 1280). `FPS` (24). Positive prompt: describe the *scene and delivery*; negative is empty by default. |
| **LTX latent audio** | nothing — encodes the speech and masks it as fixed |
| **Prepare LTX Latent** | nothing — frame count is computed from the audio (`fps × seconds + 1`) |
| **Sampler – Second Pass** | nothing — `ManualSigmas` 0.909 / 0.725 / 0.42 / 0 refine after 2× upscale |
| **Output** | two `Video Combine` nodes: the upscaled one saves as `video/LTX-2_xxxxx-audio.mp4` with the speech track |

**Using audio you already generated** (from `voice_explorer.ipynb`): bypass the whole *Qwen TTS*
group, add a `LoadAudio` for `input_audio/<line>.wav`, and wire it to both `LTXVAudioVAEEncode.audio`
and `Audio Duration (mtb).audio`. Dia gasps, Qwen-designed voices, F5 clones — all work the same way.

**Length:** LTX-2 handles ~5–20 s in one shot. Past ~20 s, split the script.

**Prompting:** Gemma reads long prompts well. Describe the person, setting, light, framing and how
they deliver the line ("speaks calmly to camera, small nods, slight smile at the end"). Don't
repeat the spoken words in the prompt — the audio already carries them.
""")

# --------------------------------------------------------------------------- #
md("---\n## 🔧 Utilities")

md("### A. Copy reference image + voice clip (or pre-made audio) into ComfyUI input")
code(r"""
import shutil, os
FILES = [
    f'{DRIVE_BASE}/input_images/my_image.png',              # reference image (update)
    f'{WAN_BASE}/voices/my_voice.wav',                      # reference voice for the in-graph clone
    # f'{WAN_BASE}/input_audio/line_01.wav',                # or a pre-generated line (bypass Qwen group)
]
for src in FILES:
    if os.path.exists(src):
        shutil.copy(src, f'/content/ComfyUI/input/{os.path.basename(src)}'); print(f'✅ Copied: {os.path.basename(src)}')
    else:
        print(f'⚠️  Not found: {src}')
""")

md(r"""
### B. Audio → frames check

The graph computes this itself; run it to sanity-check length before queuing (24 fps, ≤ ~20 s).
""")
code(r"""
import subprocess, math
AUDIO = f'{WAN_BASE}/input_audio/line_01.wav'   # update
FPS = 24
dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', AUDIO],
                           capture_output=True, text=True).stdout)
frames = int(FPS * round(dur)) + 1
print(f'{dur:.2f}s → {frames} frames @ {FPS}fps' + ('   ⚠️  over ~20s — split the line' if dur > 20 else ''))
""")

md(r"""
### C. Last frame / chaining

Output goes to `ComfyUI_Wan/output/video/`, so the **LastFrame watcher on your PC** handles this
automatically (720×1280 PNG into `ComfyUI_Qwen/last_frames`). Manual fallback:
""")
code(r"""
import shutil
CLIP_IN    = f'{WAN_BASE}/output/video/LTX-2_00001-audio.mp4'   # update
LAST_FRAME = f'{DRIVE_BASE}/input_images/last_frame.png'
!ffmpeg -y -loglevel error -sseof -1 -i "{CLIP_IN}" -frames:v 1 "{LAST_FRAME}"
shutil.copy(LAST_FRAME, '/content/ComfyUI/input/last_frame.png')
print('✅ Last frame extracted and ready in ComfyUI input')
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 📋 Quick Reference

**When to use which S2V engine**
- Tight lip-sync on a short talking line → try **Wan S2V** first (trained for it).
- A 10–20 s line in one pass, or a prompt that needs real scene description → **LTX-2**.
- Same audio through both and compare — they share `voices/` and `input_audio/`.

**Colab GPUs** — no multi-GPU runtimes exist, and ComfyUI wouldn't use a second card anyway.
What matters is one big card: Pro+ offers **A100-80GB**, **H100 (80GB)** and the **G4 / RTX PRO 6000 (96GB)**;
availability varies by session. A100-40GB is the floor for this notebook.

### Troubleshooting
- **Cloudflare "authentication" / 403** → use `TUNNEL = "colab"` in Step 6 (no auth, no external service). The Cloudflare path also wipes stale `~/.cloudflared` creds, which are the usual cause.
- **Cell stops immediately** → Step 6 prints the ComfyUI log tail on failure; read it for the real error.
- **OOM on a 40GB card** → in the *I2V* group set `WIDTH`/`HEIGHT` to 512×896 (the second pass still upscales 2×), keep lines ≤ 10 s, and/or relaunch Step 6 with `EXTRA_ARGS = ['--lowvram']`.
- **`LatentUpscaleModelLoader` dropdown empty** → file must be in `models/latent_upscale_models/` (Step 4 links it); click *Refresh* in ComfyUI.
- **Missing LTX nodes (`LTXVImgToVideoInplace`, `LTXVEmptyLatentAudio`…)** → ComfyUI too old; re-run Step 3.
- **TD Qwen loader shows no models** → check `models/Qwen3-TTS-Models` symlink (Step 4) and the Qwen download (Step 5).
- **Video ignores the audio (mouth doesn't move)** → the audio mask must be all-zero (`SolidMask` value 0) and the *LTX latent audio* group must not be bypassed.
- **Speech clipped at the end** → `Audio Duration (mtb)` rounds seconds; pad the line with a short pause or add 1 s of silence.
- **comfy_mtb install errors** → it has heavy optional deps; the two nodes used here (Audio Duration, Audio To Text) need only the base requirements. Re-run Step 3.
- **Newer LTX**: LTX-2.3 (22B) and 2.5 exist with Kijai repacks (`Kijai/LTX2.3_comfy`); this graph targets 2.0 and its node/VAE set. Treat an upgrade as a new workflow, not a model swap.
""")

# --------------------------------------------------------------------------- #
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "machine_shape": "hm", "gpuType": "A100"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
out = sys.argv[1]
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"wrote {out}: {len(cells)} cells")

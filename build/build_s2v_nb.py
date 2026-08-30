# -*- coding: utf-8 -*-
"""Generate wan22_s2v.ipynb (Colab) mirroring the structure of wan22_i2v.ipynb."""
import json
import sys

cells = []

# Shared launch cell (see launch_cell.py): same-origin Colab proxy / ngrok / cloudflare.
import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location("launch_cell", _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "launch_cell.py"))
_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
LAUNCH_CELL = _mod.LAUNCH_CELL


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n")})


# --------------------------------------------------------------------------- #
md(r"""
# 🎤 Wan 2.2 Speech-to-Video (S2V) + Voice Cloning — ComfyUI on Colab Pro

This is a **separate track** from your `wan22_i2v` notebook, not a replacement. It runs in the
opposite direction from what you might expect:

> **You don't generate a video and then add speech. You generate the speech first, and the audio drives the video.**

```
 reference voice clip (3–15 s)  +  your script text
              │
              ▼
      F5-TTS zero-shot clone  ──►  speech.wav
                                       │
 reference image  +  text prompt  +  speech.wav
              │
              ▼
        Wan 2.2 S2V (one 14B model)  ──►  talking video, lip-synced, with audio track
```

S2V is still image-to-video — the reference image is a **required** input — but it's a specialist
tuned for talking, singing and gesturing. Keep using your I2V graph for anything without dialogue.

| Setting | Value |
|---|---|
| S2V model | `wan2.2_s2v_14B_fp8_scaled.safetensors` (~16.4GB) — **one** model, no high/low pair |
| Audio encoder | `wav2vec2_large_english_fp16.safetensors` (~631MB) → `models/audio_encoders/` |
| Text encoder | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` — **shared with I2V, already in Drive** |
| VAE | `wan_2.1_vae.safetensors` — **shared with I2V, already in Drive** |
| Speed LoRA (optional) | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors` (~1.2GB) → 4 steps @ cfg 1.0 |
| Voice cloning | F5-TTS custom node (zero-shot, no training) |
| Frame chunking | 77 frames (~4.8 s) per segment, chained internally → minute-long clips possible |
| Expected VRAM | ~25–30GB on A100 (fp8) |

**Requirements:** Colab Pro · A100 GPU · ~20GB more free Drive space on top of the I2V setup

`Runtime > Change runtime type > A100 GPU`
""")

# --------------------------------------------------------------------------- #
md("## Step 1 — Verify GPU")
code(r"""
import subprocess
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(f'GPU: {gpu}')
if 'A100' in gpu:
    print('✅ A100 confirmed — fp8 S2V + 2–3 extension chunks is comfortable')
elif 'L4' in gpu:
    print('⚠️  L4 — S2V fp8 is tight; keep to 1 chunk (77 frames) and 480p')
elif 'T4' in gpu:
    print('⚠️  T4 (16 GB) is too small for 14B fp8. Switch to A100: Runtime → Change runtime type → A100 GPU')
else:
    print('⚠️  Recommended: A100. Runtime → Change runtime type → A100 GPU')
""")

# --------------------------------------------------------------------------- #
md(r"""
## Step 2 — Mount Google Drive

Uses the **same `ComfyUI_Wan` folder** as the I2V notebook so the text encoder, VAE and LoRAs are
shared. Adds three folders: `models/audio_encoders`, `voices` (reference clips for cloning) and
`input_audio` (generated speech / cleaned audio).
""")
code(r"""
from google.colab import drive
import os

drive.mount('/content/drive')

DRIVE_BASE = '/content/drive/MyDrive/ComfyUI_Wan'
MODELS_DIR = f'{DRIVE_BASE}/models'

for d in [f'{MODELS_DIR}/diffusion_models', f'{MODELS_DIR}/text_encoders',
          f'{MODELS_DIR}/vae', f'{MODELS_DIR}/clip_vision', f'{MODELS_DIR}/loras',
          f'{MODELS_DIR}/audio_encoders',      # wav2vec2 (S2V audio encoder)
          f'{MODELS_DIR}/checkpoints',         # F5-TTS caches its models under checkpoints/F5-TTS
          f'{MODELS_DIR}/qwen-tts',            # Qwen3-TTS checkpoints
          f'{DRIVE_BASE}/output', f'{DRIVE_BASE}/input_images',
          f'{DRIVE_BASE}/voices',              # reference voice clips (.wav + matching .txt transcript)
          f'{DRIVE_BASE}/input_audio']:        # generated speech / cleaned vocals that drive S2V
    os.makedirs(d, exist_ok=True)

print(f'✅ Drive mounted: {DRIVE_BASE}')
""")

# --------------------------------------------------------------------------- #
md(r"""
## Step 3 — Install ComfyUI + Custom Nodes

Same as I2V plus **ComfyUI-F5-TTS** for zero-shot voice cloning. S2V nodes
(`AudioEncoderLoader`, `AudioEncoderEncode`, `WanSoundImageToVideo`) are built into ComfyUI —
this step does a `git pull` so you're on a version that has them.
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
                   ('ComfyUI-F5-TTS', 'https://github.com/niknah/ComfyUI-F5-TTS.git'),        # zero-shot voice cloning
                   ('ComfyUI-Qwen-TTS', 'https://github.com/flybirdxx/ComfyUI-Qwen-TTS.git'),  # voice design / presets / clone / dialogue
                   ('rgthree-comfy', 'https://github.com/rgthree/rgthree-comfy.git')]:       # group bypasser used by the Qwen workflow
    path = f'/content/ComfyUI/custom_nodes/{name}'
    if not os.path.exists(path):
        !git clone -q --recursive {repo} {path}
        if os.path.exists(f'{path}/requirements.txt'):
            !pip install -q -r {path}/requirements.txt
        print(f'✅ {name} installed')
    else:
        !cd {path} && git pull -q
        print(f'✅ {name} ready')

# F5-TTS ships its engine as a git submodule; clone it manually if it came in empty
f5 = '/content/ComfyUI/custom_nodes/ComfyUI-F5-TTS/F5-TTS'
if os.path.isdir(f5) and not os.listdir(f5):
    !rm -rf {f5} && git clone -q https://github.com/SWivid/F5-TTS.git {f5}
    print('✅ F5-TTS engine cloned')

# Qwen3-TTS is incompatible with transformers 5.x — pin the version its README requires
!pip install -q "transformers==4.57.3"

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
         for f in ['diffusion_models', 'text_encoders', 'vae', 'clip_vision', 'loras',
                   'audio_encoders', 'checkpoints', 'qwen-tts']}
links['/content/ComfyUI/output'] = f'{DRIVE_BASE}/output'

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
## Step 5 — Download S2V Models (First Run Only ~20 min)

Only the S2V-specific files are new (~18GB). The text encoder and VAE are shared with the I2V
notebook and will be skipped if they're already in Drive.
""")
code(r"""
import os
HF22 = 'https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files'
HF21 = 'https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files'

models = [
    # --- S2V-specific (new downloads) ---
    ('Wan 2.2 S2V 14B fp8 (~16.4GB)',
     f'{HF22}/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors',
     f'{MODELS_DIR}/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors'),
    ('wav2vec2 audio encoder (~631MB)',
     f'{HF22}/audio_encoders/wav2vec2_large_english_fp16.safetensors',
     f'{MODELS_DIR}/audio_encoders/wav2vec2_large_english_fp16.safetensors'),
    # The official S2V template uses the T2V v1.1 high-noise lightning LoRA (not the I2V one).
    ('Lightning 4-step LoRA — T2V v1.1 high noise (~1.2GB)',
     f'{HF22}/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors',
     f'{MODELS_DIR}/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors'),

    # --- Shared with I2V — skip if already downloaded ---
    ('Text Encoder UMT5 fp8 (~6GB)',
     f'{HF21}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors',
     f'{MODELS_DIR}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors'),
    ('VAE — wan_2.1_vae (~242MB)',
     f'{HF21}/vae/wan_2.1_vae.safetensors',
     f'{MODELS_DIR}/vae/wan_2.1_vae.safetensors'),
]

for label, url, dest in models:
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f'  ✅ Present ({os.path.getsize(dest)/1024**3:.1f}GB): {label}')
    else:
        print(f'  ⬇️  Downloading: {label}')
        !wget -q --show-progress -O "{dest}" "{url}"
        print(f'  ✅ Done ({os.path.getsize(dest)/1024**3:.1f}GB): {label}')
print('\n✅ All models ready')
""")

md(r"""
### Step 5b — Qwen3-TTS checkpoints (optional, ~8 GB, first run only)

For the *Qwen3TTS Voice Generation Collection* workflow. Downloaded straight into Drive under
`models/qwen-tts/Qwen/…`, which is where the nodes look. Skip any you won't use:

| Checkpoint | Used by | Size |
|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | Voice Design group (describe a voice in prose) | ~3.5 GB |
| `Qwen3-TTS-12Hz-1.7B-Base` | Voice Clone group + clone-version Dialogue | ~3.5 GB |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | Custom Voice group (presets: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee) | ~3.5 GB |
| `Qwen3-TTS-Tokenizer-12Hz` | all of them | small |
""")
code(r"""
import os
from huggingface_hub import snapshot_download

QWEN_MODELS = [
    'Qwen3-TTS-Tokenizer-12Hz',
    'Qwen3-TTS-12Hz-1.7B-VoiceDesign',
    'Qwen3-TTS-12Hz-1.7B-Base',
    'Qwen3-TTS-12Hz-1.7B-CustomVoice',
]
for name in QWEN_MODELS:
    dest = f'{MODELS_DIR}/qwen-tts/Qwen/{name}'
    if os.path.exists(f'{dest}/config.json') or os.path.exists(f'{dest}/model.safetensors'):
        print(f'  ✅ Already present: {name}')
        continue
    print(f'  ⬇️  Downloading: {name}')
    snapshot_download(repo_id=f'Qwen/{name}', local_dir=dest)
    print(f'  ✅ Done: {name}')
print('\n✅ Qwen3-TTS checkpoints ready')
""")

# --------------------------------------------------------------------------- #
md(r"""
## Step 6 — Launch ComfyUI + public URL

**Keep this cell running.** `TUNNEL = "colab"` (default) opens ComfyUI same-origin via Colab's
own port proxy — no 403 host/origin errors (ComfyUI v1.19+), no incognito dance. Use `ngrok` if you
need a URL you can open elsewhere; `cloudflare` is the fallback.

Once it's open: **Workflow → Browse Templates → search "Wan2.2 S2V"** — see the section below
for what to set on each node.
""")
code(LAUNCH_CELL)

# --------------------------------------------------------------------------- #
md(r"""
---
## 🎛️ The S2V graph — what to set on each node

Open **Workflow → Browse Templates → "Wan2.2 S2V"**. It's the same skeleton as your I2V graph with
one extra audio branch and **one** model instead of two. Nothing about the high/low split, the
0.875 handoff, or dual LoRA slots applies here.

| Node | Setting |
|---|---|
| **Load Diffusion Model** (`UNETLoader`) | `wan2.2_s2v_14B_fp8_scaled.safetensors`, weight dtype `default` |
| **Load CLIP** (`CLIPLoader`) | `umt5_xxl_fp8_e4m3fn_scaled.safetensors`, type `wan` |
| **Load VAE** (`VAELoader`) | `wan_2.1_vae.safetensors` |
| **Load Audio Encoder** (`AudioEncoderLoader`) | `wav2vec2_large_english_fp16.safetensors` |
| **Load Audio** (`LoadAudio`) | your speech `.wav` (from the F5-TTS step or `input_audio/`) |
| **Load Image** (`LoadImage`) | the reference image — required, exactly like I2V |
| **LoraLoaderModelOnly** (optional) | `wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors`, strength 1.0 |
| **ModelSamplingSD3** | shift `8` (template default) |
| **KSampler** — with lightning LoRA | steps `4`, cfg `1.0`, `euler` / `simple` |
| **KSampler** — without LoRA | steps `20`, cfg `6.0` |
| **WanSoundImageToVideo** | width/height (e.g. `640x640` or `480x832` portrait), chunk length `77`, `length` = frames per chunk |
| **Video S2V Extend** subgraphs | one per extra 77-frame chunk (see length formula) |
| **Save Video / Video Combine** | fps `16`; VHS `Video Combine` with the audio wired in gives you the speech track baked into the MP4 |

**How long will it be?** The model runs at 16 fps in 77-frame chunks (~4.8 s each):

```
total_frames  = audio_seconds × 16
chunks_needed = ceil(total_frames / 77)
extend nodes  = chunks_needed − 1
batch size    = chunks_needed        (the template sets this automatically)
```

A 14 s line → 224 frames → 3 chunks → 2 *Extend* subgraph nodes. The "Audio → frames calculator"
utility below does this for you.

**Prompting:** the text prompt still steers action and environment ("looks at camera, slight smile,
soft studio light"). Timing, expression and gesture rhythm come from the audio. Don't feed it
silence — that's asking a speech-trained model to work without its main signal.
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 🗣️ Qwen3-TTS in the graph — design a voice in prose, style per line, dialogue

The workflow **`ComfyUI_Wan/Qwen3TTS_Voice_Generation_Collection_V3.json`** (FaboroHacks) is on
Drive. Load it with **Workflow → Open** (or drag it onto the canvas). It has five groups — enable
one at a time with the *Fast Groups Bypasser* node:

| Group | Node | Use it for |
|---|---|---|
| **Voice Design** | `FB_Qwen3TTSVoiceDesign` | text + a **written description** of the voice → speech. No reference clip needed. |
| Custom Voice | `FB_Qwen3TTSCustomVoice` | preset speakers + an `instruct` line ("incredulous, panic creeping in") |
| Voice Clone | `FB_Qwen3TTSVoiceClone` | reference `.wav` (+ transcript, or tick `x_vector_only`) → your text in that voice |
| Dialogue (design) | `RoleBank` + `DialogueInference` | multi-character script, `Name: line` per row, each role a designed voice |
| Dialogue (clone) | same + `VoiceClonePrompt` | same, each role cloned from a clip |

**Wiring it into S2V (no file round-trip):** delete/bypass the group's `SaveAudio`, drag the
Qwen node's `audio` output into `AudioEncoderEncode.audio` in the S2V graph, and queue. Everything
else in the S2V graph stays the same. If you'd rather keep the audio, leave `SaveAudio` on —
it writes to `output/audio/` on Drive, which `LoadAudio` can read back later.

**Tips**
- Voice descriptions work best when they read like casting notes: age, register, timbre, pacing,
  breath, mood. The workflow's *Test Inputs* note has three good examples to riff on.
- `instruct` is per-line emotion control — the thing F5 doesn't have. Keep it to one or two traits.
- `language` = `Auto` is fine for English; set it explicitly for mixed-language scripts.
- Same seed + same description = same voice. Pin the seed once you find one, or better: save the
  design output as `voices/<name>.wav` and clone it from then on (F5 or Qwen) for consistency.
- Non-verbal sounds aren't a documented feature — write them as words ("*gasp* — what?") or use Dia.
- Prefers `transformers==4.57.3` (Step 3 pins it). If a Qwen node errors after an update,
  right-click → *Reload node*, as the workflow's own note suggests.
""")

md("---\n## 🔧 Utilities")

md(r"""
### A. Prepare a reference voice clip (for cloning)

F5-TTS needs **3–15 s of clean speech** — one speaker, no music, no background noise — plus a
transcript of exactly what's said in it. It hard-cuts at 15 s, so trim first.

Upload the raw clip to `Drive > ComfyUI_Wan > voices/`, then run this cell to trim it, strip any
music with Demucs, and convert to a clean 24 kHz mono WAV.
""")
code(r"""
import os, subprocess, shutil

RAW_CLIP   = f'{DRIVE_BASE}/voices/raw_voice.mp3'   # update — any format ffmpeg reads
VOICE_NAME = 'my_voice'                              # becomes my_voice.wav + my_voice.txt
START_SEC  = 0                                       # where to start the excerpt
CLIP_SEC   = 12                                      # keep ≤ 15
SEPARATE_VOCALS = True                               # set False if the clip is already clean speech

work = '/tmp/voice_prep'
shutil.rmtree(work, ignore_errors=True); os.makedirs(work)
trimmed = f'{work}/trimmed.wav'
!ffmpeg -y -loglevel error -ss {START_SEC} -t {CLIP_SEC} -i "{RAW_CLIP}" -ac 1 -ar 44100 "{trimmed}"

src = trimmed
if SEPARATE_VOCALS:
    if subprocess.run(['which', 'demucs'], capture_output=True).returncode != 0:
        !pip install -q demucs
    !demucs -q --two-stems=vocals -o {work} "{trimmed}"
    src = f'{work}/htdemucs/trimmed/vocals.wav'
    print('✅ Vocals separated')

final = f'{DRIVE_BASE}/voices/{VOICE_NAME}.wav'
!ffmpeg -y -loglevel error -i "{src}" -ac 1 -ar 24000 "{final}"
print(f'✅ Reference clip ready: {final}')

txt = f'{DRIVE_BASE}/voices/{VOICE_NAME}.txt'
if not os.path.exists(txt):
    open(txt, 'w').write('TYPE THE EXACT WORDS SPOKEN IN THE CLIP HERE')
    print(f'⚠️  Now edit the transcript: {txt}')
else:
    print(f'✅ Transcript present: {txt}')
""")

md(r"""
### B. Generate speech in the cloned voice (F5-TTS)

Two ways to do this — pick one:

**In the graph:** run the cell below to copy the voice clip + transcript into `ComfyUI/input`, then add an
**F5-TTS Audio** node (`F5TTSAudio`), pick `my_voice.wav` as the sample, type your script, and wire
its `AUDIO` output straight into `AudioEncoderEncode` — no file round-trip.

**In the notebook:** the second cell runs F5-TTS's CLI and saves `input_audio/<name>.wav` to Drive,
which you then load with `LoadAudio`. Handy when you want to audition several takes before spending
GPU time on video.
""")
code(r"""
import shutil, os
VOICE_NAME = 'my_voice'      # or a library voice, e.g. 'library/libri_1272'
for ext in ('.wav', '.txt'):
    src = f'{DRIVE_BASE}/voices/{VOICE_NAME}{ext}'
    if os.path.exists(src):
        shutil.copy(src, f'/content/ComfyUI/input/{os.path.basename(VOICE_NAME)}{ext}')
        print(f'✅ Copied {os.path.basename(VOICE_NAME)}{ext} → ComfyUI/input')
    else:
        print(f'⚠️  Missing: {src}')
""")
code(r"""
import os, subprocess

VOICE_NAME = 'my_voice'
SCRIPT     = "Hey, thanks for watching. Today I want to show you something I've been working on."
OUT_NAME   = 'line_01'          # → input_audio/line_01.wav

ref_wav = f'{DRIVE_BASE}/voices/{VOICE_NAME}.wav'
txt_path = f'{DRIVE_BASE}/voices/{VOICE_NAME}.txt'
ref_txt = open(txt_path).read().strip() if os.path.exists(txt_path) else ''
if not ref_txt or ref_txt.startswith('TYPE THE EXACT'):
    ref_txt = ''   # empty → F5-TTS auto-transcribes the reference with Whisper (slower first run, fine for tests)
    print('ℹ️  No transcript — F5-TTS will auto-transcribe the reference clip')
out_dir = f'{DRIVE_BASE}/input_audio'

if subprocess.run(['which', 'f5-tts_infer-cli'], capture_output=True).returncode != 0:
    !pip install -q f5-tts
!f5-tts_infer-cli --model F5TTS_v1_Base --ref_audio "{ref_wav}" --ref_text "{ref_txt}" --gen_text "{SCRIPT}" --output_dir "{out_dir}" --output_file "{OUT_NAME}.wav"

out = f'{out_dir}/{OUT_NAME}.wav'
if os.path.exists(out):
    import shutil; shutil.copy(out, f'/content/ComfyUI/input/{OUT_NAME}.wav')
    dur = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                          '-of', 'csv=p=0', out], capture_output=True, text=True).stdout.strip()
    print(f'✅ Speech saved: {out}  ({float(dur):.1f}s) — also copied to ComfyUI/input')
else:
    print('⚠️  No output produced — check the log above')
""")

md(r"""
### C. Audio → frames calculator

Paste the audio you're about to use and this tells you how many *Extend* nodes to add.
""")
code(r"""
import subprocess, math
AUDIO = f'{DRIVE_BASE}/input_audio/line_01.wav'   # update
FPS, CHUNK = 16, 77

dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                            '-of', 'csv=p=0', AUDIO], capture_output=True, text=True).stdout)
frames = math.ceil(dur * FPS)
chunks = math.ceil(frames / CHUNK)
print(f'Audio     : {dur:.2f}s')
print(f'Frames    : {frames} @ {FPS}fps')
print(f'Chunks    : {chunks} × {CHUNK} frames  ({chunks * CHUNK / FPS:.1f}s of video)')
print(f'Extend nodes to add : {chunks - 1}')
print(f'Batch size          : {chunks}')
""")

md(r"""
### D. Clean up an existing audio track (strip music / noise)

ComfyUI has no vocal separation built in, and mixed audio confuses the lip-sync badly. Run any
audio that isn't pure speech through this before using it to drive S2V.
""")
code(r"""
import os, subprocess, shutil
SOURCE   = f'{DRIVE_BASE}/input_audio/raw_track.mp3'   # update
OUT_NAME = 'raw_track_vocals'

if subprocess.run(['which', 'demucs'], capture_output=True).returncode != 0:
    !pip install -q demucs
work = '/tmp/demucs_out'
shutil.rmtree(work, ignore_errors=True)
!demucs -q --two-stems=vocals -o {work} "{SOURCE}"
stem = os.path.splitext(os.path.basename(SOURCE))[0]
vocals = f'{work}/htdemucs/{stem}/vocals.wav'
final = f'{DRIVE_BASE}/input_audio/{OUT_NAME}.wav'
!ffmpeg -y -loglevel error -i "{vocals}" -ac 1 -ar 24000 "{final}"
shutil.copy(final, f'/content/ComfyUI/input/{OUT_NAME}.wav')
print(f'✅ Vocals only: {final} — also copied to ComfyUI/input')
""")

md("### E. Copy reference image + audio into ComfyUI input")
code(r"""
import shutil, os
FILES = [
    f'{DRIVE_BASE}/input_images/my_image.jpg',   # update
    f'{DRIVE_BASE}/input_audio/line_01.wav',     # update
]
for src in FILES:
    if os.path.exists(src):
        shutil.copy(src, f'/content/ComfyUI/input/{os.path.basename(src)}')
        print(f'✅ Copied: {os.path.basename(src)}')
    else:
        print(f'⚠️  Not found: {src}')
""")

md(r"""
### F. Extract last frame (for chaining clips)

S2V output lands in `ComfyUI_Wan/output/video/` like everything else, so the **LastFrame watcher on
your PC** picks it up automatically. This cell is the manual fallback.
""")
code(r"""
import shutil
CLIP_IN    = f'{DRIVE_BASE}/output/video/ComfyUI_00001_.mp4'   # update filename
LAST_FRAME = f'{DRIVE_BASE}/input_images/last_frame.jpg'
!ffmpeg -y -loglevel error -sseof -1 -i "{CLIP_IN}" -frames:v 1 "{LAST_FRAME}"
shutil.copy(LAST_FRAME, '/content/ComfyUI/input/last_frame.jpg')
print('✅ Last frame extracted and ready in ComfyUI input')
""")

md(r"""
### G. Mux audio onto a silent video (lip-sync-after-the-fact fallback)

If you re-animate an existing I2V clip with a lip-sync model instead of using S2V, or your Save Video
node dropped the audio, this puts the track back.
""")
code(r"""
VIDEO  = f'{DRIVE_BASE}/output/video/ComfyUI_00001_.mp4'   # update
AUDIO  = f'{DRIVE_BASE}/input_audio/line_01.wav'           # update
OUTPUT = f'{DRIVE_BASE}/output/video/ComfyUI_00001_with_audio.mp4'
!ffmpeg -y -loglevel error -i "{VIDEO}" -i "{AUDIO}" -c:v copy -c:a aac -shortest "{OUTPUT}"
print(f'✅ Saved: {OUTPUT}')
""")

md(r"""
### H. Test voices — no recording needed

Two free sources of clean, single-speaker reference clips, so you can test the whole pipeline
before you bother recording anyone.

**H1 — F5-TTS's bundled demo voices.** Installed with the node in Step 3. `f5_demo` comes with its
transcript; the three story voices (`f5_main`, `f5_town`, `f5_country`) don't, so use the CLI route
(Utility B) for those — it auto-transcribes when the `.txt` is empty.

**H2 → H4 — a browsable voice library from LibriTTS-R** (CC BY 4.0, 24 kHz restored audiobook
speech). Three cells:

- **H2 builds the library**: one pass over a split, collecting ~10 s + a transcript for *every*
  speaker in it, labelled with gender / pitch / speaking rate / accent / expressiveness from the
  Parler-TTS annotations. `dev.clean` = 40 voices in ~3 min; add `test.clean` for ~79. Saved to
  `voices/library/libri_<id>.wav` + `.txt` and a `catalog.csv`, so you only ever run it once.
- **H3 browses it**: filter by keyword, get a table plus an audio player per voice.
- **H4 auditions**: clones your actual line in several candidate voices at once so you're comparing
  results, not references.

These are audiobook narrators (LibriVox volunteers), so expect a "read aloud" delivery. Attribution
(CC BY) applies if you publish. Use a library voice anywhere as `VOICE_NAME = 'library/libri_<id>'`.
""")
code(r"""
# H1 — copy F5-TTS's bundled example voices into Drive > voices/
import glob, os, shutil, subprocess

candidates = glob.glob('/content/ComfyUI/custom_nodes/ComfyUI-F5-TTS/**/infer/examples', recursive=True)
if not candidates:
    import importlib.util
    spec = importlib.util.find_spec('f5_tts')
    if spec and spec.submodule_search_locations:
        candidates = glob.glob(f'{list(spec.submodule_search_locations)[0]}/infer/examples')
if not candidates:
    raise SystemExit('⚠️  F5-TTS examples not found — run Step 3 first')
ex = candidates[0]

demos = {
    # name        : (source file,                  known transcript or '' for auto-transcribe)
    'f5_demo'     : (f'{ex}/basic/basic_ref_en.wav', 'Some call me nature, others call me mother nature.'),
    'f5_main'     : (f'{ex}/multi/main.flac',        ''),
    'f5_town'     : (f'{ex}/multi/town.flac',        ''),
    'f5_country'  : (f'{ex}/multi/country.flac',     ''),
}
for name, (src, text) in demos.items():
    if not os.path.exists(src):
        print(f'⚠️  missing {src}'); continue
    dst = f'{DRIVE_BASE}/voices/{name}.wav'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-ac', '1', '-ar', '24000', dst], check=True)
    open(f'{DRIVE_BASE}/voices/{name}.txt', 'w').write(text)
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', dst],
                               capture_output=True, text=True).stdout)
    print(f'✅ {name}.wav  ({dur:.1f}s)  transcript: {"yes" if text else "auto"}')
print('\nUse any of these as VOICE_NAME in Utility B.')
""")
code(r"""
# H2 — build the voice library: ~10 s + transcript for every speaker in the chosen split(s), with labels
import io, os, csv, collections, time
import numpy as np, soundfile as sf
from datasets import load_dataset, Audio

SPLITS       = ['dev.clean']    # 40 voices. Add 'test.clean' (+39). 'train.clean.100' adds ~250 but takes ~20 min.
TARGET_SEC   = 10               # seconds of speech to collect per speaker (hard cap 15)
MAX_SPEAKERS = None             # e.g. 12 to stop early while testing

LIB = f'{DRIVE_BASE}/voices/library'
os.makedirs(LIB, exist_ok=True)
SR = 24000
gap = np.zeros(int(0.3 * SR), dtype=np.float32)

# --- 1. speaker labels from the Parler-TTS annotations (tiny, metadata only) ---
def mode(values):
    values = [v for v in values if v]
    return collections.Counter(values).most_common(1)[0][0] if values else '?'

per_spk = collections.defaultdict(list)
for split in SPLITS:
    meta = load_dataset('parler-tts/libritts-r-filtered-speaker-descriptions', 'clean', split=split)
    for r in meta.select_columns(['speaker_id', 'gender', 'pitch', 'speaking_rate', 'accent', 'speech_monotony']):
        per_spk[r['speaker_id']].append(r)
LABEL_KEYS = ('gender', 'pitch', 'speaking_rate', 'accent', 'speech_monotony')
labels = {spk: {k: mode([r[k] for r in rs]) for k in LABEL_KEYS} for spk, rs in per_spk.items()}
print(f'Labels for {len(labels)} speakers')

# --- 2. one streaming pass over the audio, decoding only what we keep ---
clips = collections.defaultdict(lambda: {'audio': [], 'text': [], 'sec': 0.0})
t0 = time.time()
for split in SPLITS:
    ds = load_dataset('mythicinfinity/libritts_r', 'clean', split=split, streaming=True)
    ds = ds.cast_column('audio', Audio(decode=False))
    for r in ds:
        spk = r['speaker_id']; c = clips[spk]
        if c['sec'] >= TARGET_SEC:
            continue
        a, sr = sf.read(io.BytesIO(r['audio']['bytes']), dtype='float32')
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != SR:
            continue
        d = len(a) / sr
        if c['sec'] + d > 15:
            continue
        c['audio'] += [a, gap]; c['text'].append(r['text_normalized'].strip()); c['sec'] += d + 0.3
        done = sum(1 for v in clips.values() if v['sec'] >= TARGET_SEC)
        if c['sec'] >= TARGET_SEC:
            print(f'  ✓ speaker {spk:>5}  {c["sec"]:4.1f}s   [{done} done, {time.time()-t0:.0f}s]')
        if MAX_SPEAKERS and done >= MAX_SPEAKERS:
            break

# --- 3. write wav + txt + catalog ---
rows = []
for spk, c in clips.items():
    if not c['audio']:
        continue
    name = f'libri_{spk}'
    sf.write(f'{LIB}/{name}.wav', np.concatenate(c['audio']), SR)
    transcript = ' '.join(c['text'])
    open(f'{LIB}/{name}.txt', 'w').write(transcript)
    lab = labels.get(spk, {k: '?' for k in LABEL_KEYS})
    rows.append({'name': name, 'speaker_id': spk, **lab, 'seconds': round(c['sec'], 1), 'transcript': transcript})

with open(f'{LIB}/catalog.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'\n✅ {len(rows)} voices saved to {LIB}  (catalog.csv alongside)')
""")
code(r"""
# H3 — browse the library: filter by keyword, listen
import pandas as pd
from IPython.display import Audio, HTML, display

GENDER  = 'any'     # 'male' / 'female' / 'any'
PITCH   = ''        # substring of: low-pitch, slightly low-pitch, moderate pitch, slightly high-pitch, high-pitch, very high-pitch
RATE    = ''        # substring of: very slowly, slowly, slightly slowly, moderate speed, slightly fast, fast, very fast
STYLE   = ''        # substring of: monotone, slightly expressive and animated, expressive and animated, very expressive
ACCENT  = ''        # e.g. 'American', 'Canadian'
SHOW    = 10        # max voices to list/play

LIB = f'{DRIVE_BASE}/voices/library'
cat = pd.read_csv(f'{LIB}/catalog.csv', dtype=str)
mask = pd.Series(True, index=cat.index)
for col, needle in (('gender', '' if GENDER == 'any' else GENDER), ('pitch', PITCH), ('speaking_rate', RATE),
                    ('speech_monotony', STYLE), ('accent', ACCENT)):
    if needle:
        mask &= cat[col].str.contains(needle, case=False, na=False)
sel = cat[mask]
print(f'{len(sel)} of {len(cat)} voices match')
display(sel[['name', 'gender', 'pitch', 'speaking_rate', 'speech_monotony', 'accent', 'seconds']].head(SHOW))

for _, r in sel.head(SHOW).iterrows():
    display(HTML(f"<b>{r['name']}</b> &nbsp;·&nbsp; {r['gender']}, {r['pitch']}, {r['speaking_rate']}, {r['speech_monotony']}"
                 f"<br><span style='color:#888'>{r['transcript'][:110]}…</span>"))
    display(Audio(f"{LIB}/{r['name']}.wav"))
""")
code(r"""
# H4 — audition: clone the SAME line in several candidate voices, side by side
import os, subprocess
from IPython.display import Audio, HTML, display

VOICES = ['library/libri_3081', 'library/libri_84', 'f5_demo']    # any names under voices/
SCRIPT = "Hey, thanks for watching. Today I want to show you something I've been working on."
SEED   = 42          # fixed seed so voices differ only by reference, not by luck

out_dir = f'{DRIVE_BASE}/input_audio'
os.makedirs(out_dir, exist_ok=True)

# Load the model once via the Python API; fall back to the CLI (reloads per voice) if the API changes.
try:
    from f5_tts.api import F5TTS
    tts = F5TTS(model='F5TTS_v1_Base')
    def synth(ref, txt, out):
        tts.infer(ref_file=ref, ref_text=txt, gen_text=SCRIPT, file_wave=out, seed=SEED, show_info=lambda *a, **k: None)
except Exception as e:
    print(f'ℹ️  F5TTS API unavailable ({type(e).__name__}) — using CLI')
    def synth(ref, txt, out):
        subprocess.run(['f5-tts_infer-cli', '--model', 'F5TTS_v1_Base', '--ref_audio', ref, '--ref_text', txt,
                        '--gen_text', SCRIPT, '--output_dir', os.path.dirname(out), '--output_file', os.path.basename(out)],
                       check=True, capture_output=True)

for v in VOICES:
    ref = f'{DRIVE_BASE}/voices/{v}.wav'
    txt_path = f'{DRIVE_BASE}/voices/{v}.txt'
    txt = open(txt_path).read().strip() if os.path.exists(txt_path) else ''
    if txt.startswith('TYPE THE EXACT'):
        txt = ''
    if not os.path.exists(ref):
        print(f'⚠️  missing {ref}'); continue
    out = f'{out_dir}/audition_{os.path.basename(v)}.wav'
    synth(ref, txt, out)
    display(HTML(f'<b>{v}</b> → {os.path.basename(out)}'))
    display(Audio(out))

print('\nPick a winner, then use it in Utility B as VOICE_NAME (e.g. "library/libri_3081").')
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 📋 Quick Reference

| | Wan 2.2 I2V (your other notebook) | Wan 2.2 S2V (this notebook) |
|---|---|---|
| Diffusion models | **2** (high + low noise) | **1** (`wan2.2_s2v_14B`) |
| Samplers | 2 KSamplerAdvanced, split at step 3 | 1 KSampler, no split |
| LoRA slots | 2 (high + low) | 1 |
| Lightning LoRA | i2v v1 high + low | `t2v_lightx2v_4steps_v1.1_high_noise` |
| Extra inputs | CLIP Vision | Audio (wav2vec2) |
| Reference image | required | required |
| Length | ≤ 161 frames in one shot | 77-frame chunks chained → minute-long |
| Tuned for | general motion, camera moves | talking, singing, gesturing |
| Audio in output | mux afterwards | baked in via Video Combine |

**When to use which:** anything where a character speaks → S2V. Everything else (orbits, action,
ambience, no dialogue) → I2V. They share the VAE and text encoder, so switching is just opening
the other template.

### Voice cloning tips
- 3–15 s reference, one speaker, no music. Demucs it if unsure (Utility A does this).
- The transcript must match the clip *exactly* — wrong words in `my_voice.txt` degrade the clone more than a slightly noisy clip does.
- Generate the speech first and listen to it. Fixing a bad take costs seconds; a bad video costs minutes.
- Other zero-shot options with ComfyUI nodes if F5 doesn't suit the voice: Qwen3-TTS (10 languages), Chatterbox, OmniVoice.

### Troubleshooting
- **Cloudflare "authentication" / 403** → use `TUNNEL = "colab"` in Step 6 (no auth, no external service). The Cloudflare path also wipes stale `~/.cloudflared` creds, which are the usual cause.
- **Cell stops immediately** → Step 6 prints the ComfyUI log tail on failure; read it for the real error.
- **`WanSoundImageToVideo` / `AudioEncoderLoader` missing** → ComfyUI too old; re-run Step 3 (it pulls latest) and restart ComfyUI
- **`wav2vec2` not in the dropdown** → check `models/audio_encoders/` symlink (Step 4) and the download (Step 5)
- **Mouth doesn't match audio** → audio has music/noise under the voice; run Utility D
- **Video shorter than audio** → not enough *Extend* nodes; run Utility C
- **OOM** → drop resolution (e.g. 480×832) before dropping chunks; fp8 model + 3 chunks fits an A100
- **F5-TTS import error about `bigvgan`** → known upstream quirk; re-run Step 3 after `rm -rf /content/ComfyUI/custom_nodes/ComfyUI-F5-TTS`, or use the CLI cell (Utility B) instead
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

# -*- coding: utf-8 -*-
"""Generate hunyuan_foley_v2a.ipynb — HunyuanVideo-Foley video-to-audio on Colab.

Runs the opposite direction from the other notebooks in this repo: video + text in,
synced 48 kHz audio out, muxed back onto the source clip."""
import json
import sys

cells = []

# Shared launch cell — same copy every ComfyUI notebook here uses.
import importlib.util as _ilu, os as _os
_spec = _ilu.spec_from_file_location(
    "build_foley_nb_launch",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "launch_cell.py"))
_mod = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_mod)
LAUNCH_CELL = _mod.LAUNCH_CELL


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n")}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": t.strip("\n")}

# ----------------------------------------------------------------- header
cells.append(md(r"""
# 🔊 HunyuanVideo-Foley — video-to-audio (ComfyUI on Colab)

Give it a **video clip + a text prompt describing the soundtrack**, get back a **synced 48 kHz audio track**. This is TV2A (text-video-to-audio): Foley and ambience, generated to land on the frames where things actually happen.

> **This notebook runs the opposite direction from the others in this repo.** `wan22_i2v`, `wan22_s2v` and `ltx2_s2v` are all *image + audio → video*. This one is *video + text → audio*, and the last step muxes the result back onto your source clip. The ingest utilities take video, not stills.

| Component | File | Size |
|---|---|---|
| Foley model (fp16) | `hunyuanvideo_foley.safetensors` | ~10.3 GB |
| Synchformer | `synchformer_state_dict_fp16.safetensors` | ~475 MB |
| DAC-VAE 48 kHz | `vae_128d_48k_fp16.safetensors` | ~743 MB |
| SigLIP2 + CLAP | pulled automatically by the Dependencies Loader | ~3 GB |

**What makes it sync:** a Synchformer-based frame-level alignment path with gated modulation, running alongside multimodal transformer blocks that process visual and audio streams together. That's the piece that puts the splash on the frame you hit the water instead of a third of a second late — the failure mode that makes generated Foley sound obviously fake.

**Setup:** `Runtime → Change runtime type → GPU`. Unlike the Wan notebooks, this one is light — ~12 GB of weights and modest VRAM. An **L4 is fine**; A100 just makes it faster.

**Storage:** weights live on the runtime's local disk and are re-pulled from Hugging Face each session (~2 min with `hf_transfer`). Only your clips and rendered audio go to Drive. Net Drive usage: your media only.

---

### ⚠️ Read this before prompting — it is the single biggest quality lever

**Write prose, not tag lists.** Training captions were generated with GenAU, which produces natural-language descriptions of audio. Full descriptive sentences match that distribution; comma-separated tags work but underperform. Write it the way a caption model would describe your clip's soundtrack.

**Name sounds, not events or emotions.** The text encoder is CLAP, trained on audio–caption pairs, so its space is organised around *what things sound like*. `"a man is thrown into a bucket of water"` underperforms `"A heavy body hits the water with a loud splash, water sloshes and displaces, wet fabric slaps, distant laughter."`

**Speech is out of scope.** This generates Foley, ambience and *non-lexical* vocalisation — screams, gasps, laughter, grunts. It will happily produce vocalisation-shaped noise that sounds like shouting without being language. For intelligible words use `voice_explorer.ipynb` / `wan22_s2v.ipynb` for TTS + lipsync, then mix the Foley underneath with utility **C**.
"""))

# ----------------------------------------------------------------- step 1
cells.append(md("## Step 1 — Verify GPU"))
cells.append(code(r"""
import subprocess
gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout.strip()
print(f'GPU: {gpu}')
if 'A100' in gpu:
    print('✅ A100 — fp16 Foley with room to spare; use PRECISION="fp16" in Step 5')
elif 'L4' in gpu:
    print('✅ L4 (24 GB) — fp16 fits. Plenty for this model.')
elif 'T4' in gpu:
    print('⚠️  T4 (16 GB) — use PRECISION="fp8_e4m3fn" in Step 5, or the xl variant.')
else:
    print('ℹ️  Any GPU with ~16 GB should manage fp16; drop to fp8 if you OOM.')
print('\nNote: Foley is far lighter than the Wan video models — you do NOT need an A100 here.')
"""))

# ----------------------------------------------------------------- step 2
cells.append(md(r"""
## Step 2 — Mount Drive & set paths

Weights go to `/content/models/foley` on the runtime's local disk — wiped when the runtime recycles, which is the point: they're re-pulled from Hugging Face instead of occupying Drive quota. Only your clips, renders and prompt logbook live in Drive.
"""))
cells.append(code(r"""
import os, shutil
from google.colab import drive

if not os.path.ismount('/content/drive'):
    drive.mount('/content/drive')

DRIVE_BASE   = '/content/drive/MyDrive/ComfyUI_Foley'

FOLEY_MODELS = '/content/models/foley'          # local: re-pulled from HF each session
INPUT_DIR    = f'{DRIVE_BASE}/input_videos'     # Drive: your source clips
OUTPUT_DIR   = f'{DRIVE_BASE}/output_audio'     # Drive: rendered audio + muxed video
PROMPTS_DIR  = f'{DRIVE_BASE}/prompts'          # Drive: your vocabulary logbook

for d in (FOLEY_MODELS, INPUT_DIR, OUTPUT_DIR, PROMPTS_DIR):
    os.makedirs(d, exist_ok=True)

free = shutil.disk_usage('/content').free / 1024**3
print('✅ Paths ready')
print(f'   foley weights (HF → local) → {FOLEY_MODELS}')
print(f'   input videos  (Drive)      → {INPUT_DIR}')
print(f'   output audio  (Drive)      → {OUTPUT_DIR}')
print(f'   prompt logbook(Drive)      → {PROMPTS_DIR}')
print(f'\n   local disk free: {free:.0f} GB   (weights need ~12 GB)')
if free < 20:
    print('⚠️  Tight. Runtime → Disconnect and delete runtime for a clean disk.')
"""))

# ----------------------------------------------------------------- step 3
cells.append(md(r"""
## Step 3 — Install ComfyUI + custom nodes

`phazei/ComfyUI-HunyuanVideo-Foley` is the node pack — it ships its own example workflow, which Step 6 installs. `if-ai/ComfyUI_HunyuanVideoFoley` is the alternative the official repo points at; both work, and on a 24 GB+ card their VRAM features (FP8, block swap, CPU offload) are irrelevant, so this picks on node ergonomics.
"""))
cells.append(code(r"""
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

if not os.path.exists('/content/comfyui_reqs_installed'):
    !pip install -q -r /content/ComfyUI/requirements.txt
    open('/content/comfyui_reqs_installed', 'w').close()
    print('✅ Requirements installed')
else:
    print('✅ Requirements already installed')

FOLEY_NODE = '/content/ComfyUI/custom_nodes/ComfyUI-HunyuanVideo-Foley'

for name, repo in [
        ('ComfyUI-Manager',            'https://github.com/ltdrdata/ComfyUI-Manager.git'),
        ('ComfyUI-VideoHelperSuite',   'https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git'),
        ('ComfyUI-HunyuanVideo-Foley', 'https://github.com/phazei/ComfyUI-HunyuanVideo-Foley.git')]:
    path = f'/content/ComfyUI/custom_nodes/{name}'
    if not os.path.exists(path):
        !git clone -q --recursive {repo} {path}
        if os.path.exists(f'{path}/requirements.txt'):
            !pip install -q -r {path}/requirements.txt
        print(f'✅ {name} installed')
    else:
        !cd {path} && git pull -q
        print(f'✅ {name} ready')

if subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode != 0:
    !apt-get install -y -q ffmpeg
print('\n✅ All installs complete')
"""))

# ----------------------------------------------------------------- step 4
cells.append(md(r"""
## Step 4 — Link storage into ComfyUI

The node pack looks for its weights in `ComfyUI/models/foley/`. That points at local disk; `input` and `output` point at Drive so your clips are visible in the Load Video node and renders persist.
"""))
cells.append(code(r"""
import os, shutil
COMFY = '/content/ComfyUI'

links = {
    f'{COMFY}/models/foley': FOLEY_MODELS,   # local disk
    f'{COMFY}/input':        INPUT_DIR,      # Drive
    f'{COMFY}/output':       OUTPUT_DIR,     # Drive
}

for dst, src in links.items():
    os.makedirs(src, exist_ok=True)
    if os.path.islink(dst):
        if os.path.realpath(dst) == os.path.realpath(src):
            print(f'  ✅ {os.path.basename(dst)} → {src}')
            continue
        os.unlink(dst)
    elif os.path.isdir(dst):
        # Rescue anything ComfyUI already wrote before replacing the directory
        for entry in os.listdir(dst):
            target = os.path.join(src, entry)
            if not os.path.exists(target):
                shutil.move(os.path.join(dst, entry), target)
        shutil.rmtree(dst)
    elif os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)
    print(f'  ✅ {os.path.basename(dst)} → {src}')
print('\n✅ Folders linked')
"""))

# ----------------------------------------------------------------- step 5
cells.append(md(r"""
## Step 5 — Download the Foley models

From [`phazei/HunyuanVideo-Foley`](https://huggingface.co/phazei/HunyuanVideo-Foley) — safetensors conversions of Tencent's original `.pth` release, which is what this node pack expects.

| `PRECISION` | Main model | Size | Use when |
|---|---|---|---|
| `fp16` | `hunyuanvideo_foley` | 10.3 GB | **Default.** 16 GB+ VRAM |
| `fp8_e4m3fn` | `hunyuanvideo_foley_fp8_e4m3fn` | 5.3 GB | Tight VRAM |
| `xl_fp16` | `hunyuanvideo_foley_xl` | 5.8 GB | Smaller variant, A/B it |
| `xl_fp8_e4m3fn` | `hunyuanvideo_foley_xl_fp8_e4m3fn` | 3.8 GB | Smallest |

> The naming is counterintuitive — **"XL" is the *smaller* checkpoint here**, not an upgrade. Don't assume it's better; A/B it on your own clip.

Synchformer and the DAC-VAE are always fetched. SigLIP2 and CLAP are **not** downloaded here — the Dependencies Loader node pulls them on first run (~3 GB, cached to `/root/.cache/huggingface`, so re-pulled each session too).

If you set `PRECISION` to an fp8 variant, set the loader node's quantization to `auto` or `fp8` in the UI, otherwise it upcasts to fp16 in memory and you lose the saving.
"""))
cells.append(code(r"""
#@title Step 5 — Download Foley models { display-mode: "form" }
PRECISION = "fp16"  #@param ["fp16", "fp8_e4m3fn", "xl_fp16", "xl_fp8_e4m3fn"]

import os, shutil

try:
    import hf_transfer  # noqa: F401
except ImportError:
    !pip install -q hf_transfer
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
from huggingface_hub import hf_hub_download

REPO = 'phazei/HunyuanVideo-Foley'
MAIN = {
    'fp16':          'hunyuanvideo_foley.safetensors',
    'fp8_e4m3fn':    'hunyuanvideo_foley_fp8_e4m3fn.safetensors',
    'xl_fp16':       'hunyuanvideo_foley_xl.safetensors',
    'xl_fp8_e4m3fn': 'hunyuanvideo_foley_xl_fp8_e4m3fn.safetensors',
}[PRECISION]

FILES = [
    (f'Foley main model ({PRECISION})', MAIN),
    ('Synchformer (temporal alignment)', 'synchformer_state_dict_fp16.safetensors'),
    ('DAC-VAE 48 kHz',                   'vae_128d_48k_fp16.safetensors'),
]

for label, fname in FILES:
    dest = f'{FOLEY_MODELS}/{fname}'
    if os.path.exists(dest) and os.path.getsize(dest) > 1024:
        print(f'  ✅ Ready ({os.path.getsize(dest)/1024**3:.2f} GB): {label}')
        continue
    print(f'  ⬇️  Downloading: {label}')
    try:
        tmp = hf_hub_download(repo_id=REPO, filename=fname, local_dir=FOLEY_MODELS)
        if os.path.abspath(tmp) != os.path.abspath(dest):
            shutil.move(tmp, dest)
    except Exception as e:
        print(f'     hf_hub_download failed ({e}); falling back to wget')
        url = f'https://huggingface.co/{REPO}/resolve/main/{fname}'
        !wget -q --show-progress -O "{dest}" "{url}"
    print(f'  ✅ Done ({os.path.getsize(dest)/1024**3:.2f} GB): {label}')

print(f'\n✅ Foley models ready at {FOLEY_MODELS}')
print('   SigLIP2 + CLAP download on first run of the Dependencies Loader node (~3 GB).')
"""))

# ----------------------------------------------------------------- step 6
cells.append(md(r"""
## Step 6 — Install the bundled workflow

The node pack ships a working example graph. Copying **its** workflow is deliberate: a hand-written ComfyUI JSON silently breaks whenever node inputs change, whereas this one is maintained alongside the nodes.
"""))
cells.append(code(r"""
import os, glob, shutil

SRC_DIR = f'{FOLEY_NODE}/example_workflows'
DST_DIR = '/content/ComfyUI/user/default/workflows'
os.makedirs(DST_DIR, exist_ok=True)

found = sorted(glob.glob(f'{SRC_DIR}/*.json'))
if not found:
    print(f'⚠️  No example workflow found in {SRC_DIR}')
    print('   Re-run Step 3, or build the chain by hand (4 nodes — see Step 8).')
else:
    for src in found:
        dst = os.path.join(DST_DIR, os.path.basename(src))
        shutil.copyfile(src, dst)
        print(f'  ✅ {os.path.basename(src)}')
    print(f'\n✅ Installed to the Workflows (📂) sidebar in ComfyUI')
"""))

# ----------------------------------------------------------------- step 7
cells.append(md(r"""
## Step 7 — Launch ComfyUI + public URL

Same tunnel cell as the other notebooks. **Keep it running.**
"""))
cells.append(code(LAUNCH_CELL))

# ----------------------------------------------------------------- step 8
cells.append(md(r"""
## Step 8 — Run it

Open the workflow from the **Workflows (📂)** sidebar. The chain is four nodes:

```
Hunyuan-Foley Model Loader
        ↓
Hunyuan-Foley Dependencies Loader      (DAC-VAE · SigLIP2 · Synchformer · CLAP)
        ↓
[Hunyuan-Foley Torch Compile]          optional, ~30% faster after first compile
        ↓
Hunyuan-Foley Sampler  ←── images + frame_rate from Load Video
        ↓
Save Audio
```

1. **Load Video** → pick a clip from `ComfyUI_Foley/input_videos/` (run utility **A** first to copy one in and check its length).
2. Connect the frames to the sampler's image input and **set `frame_rate` to the clip's real fps**. Getting this wrong shifts every sound in the track — it is the most common cause of "the audio is close but drifting."
3. Prompt: **prose, sounds not events.** See utility **B**.
4. **Euler · CFG 4.5 · 50 steps** to start.
5. Queue.

### Two-pass layering — the thing that actually makes it sound good

A single prompt covering both the physical event and the vocal reaction usually lets one swamp the other. Run the clip **twice** and mix:

| Pass | Prompt |
|---|---|
| **Physical** | `A heavy body hits the water with a loud splash. Water sloshes and displaces against the sides. Wet fabric slaps.` |
| **Vocal** | `A man lets out a loud startled scream, followed by distant laughter.` |

Same video into both, so Synchformer aligns both passes to the same visual onsets. Then utility **C** loudness-matches them and mixes at a balance you choose, instead of hoping the model's internal weighting matches your taste.

Costs you double the inference time. Worth it.

### Free vocabulary probe

**The sampler's image input is optional** — leave it unconnected and the node runs as pure text-to-audio. That's a couple of seconds per phrase with no video encoding pass, which makes it a fast way to test whether a phrasing is in distribution before committing it to a real render. Log the winners with utility **B**.
"""))

# ----------------------------------------------------------------- utilities
cells.append(md("---\n## 🔧 Utilities"))

# --- A
cells.append(md(r"""
### A. Ingest a clip — and check its length

Copies a clip from Drive into ComfyUI's input and prints what the sampler needs to know. **`fps` is the number you type into `frame_rate`.**

Foley is trained on short windows; long clips drift and cost memory. This warns past `SEGMENT_SECONDS` and offers a split. The exact ceiling depends on your config and VRAM — treat the threshold as a starting guess and raise it if your clips hold up.
"""))
cells.append(code(r"""
#@title A. Ingest a clip { display-mode: "form" }
CLIP = "my_clip.mp4"      #@param {type:"string"}
SEGMENT_SECONDS = 10      #@param {type:"number"}
SPLIT_IF_LONG = False     #@param {type:"boolean"}

import os, json, subprocess, math

src = CLIP if os.path.isabs(CLIP) else f'{INPUT_DIR}/{CLIP}'
if not os.path.exists(src):
    print(f'⚠️  Not found: {src}')
    have = [f for f in sorted(os.listdir(INPUT_DIR)) if f.lower().endswith(('.mp4','.mov','.webm','.mkv','.avi'))]
    print('   In input_videos/:', ', '.join(have) if have else '(empty — upload a clip to Drive)')
else:
    # NOTE: ffprobe honours only the LAST -show_entries, so stream and format
    # sections must be combined into a single argument separated by ':'.
    probe = subprocess.run(
        ['ffprobe','-v','error','-select_streams','v:0','-of','json',
         '-show_entries','stream=r_frame_rate,nb_frames,width,height:format=duration',
         src],
        capture_output=True, text=True).stdout
    info = json.loads(probe or '{}')
    if not info.get('streams'):
        raise SystemExit(f'ffprobe found no video stream in {src}')
    st   = info['streams'][0]
    num, den = (st['r_frame_rate'].split('/') + ['1'])[:2]
    fps  = float(num) / float(den or 1)
    dur  = float(info['format']['duration'])
    frames = int(st.get('nb_frames') or round(dur * fps))

    print(f'  file       {os.path.basename(src)}')
    print(f'  resolution {st["width"]}x{st["height"]}')
    print(f'  fps        {fps:.3f}   ← type this into the sampler\'s frame_rate')
    print(f'  duration   {dur:.2f} s')
    print(f'  frames     {frames}')

    # /content/ComfyUI/input is symlinked to INPUT_DIR, so the clip is already visible.
    print(f'\n✅ Visible in the Load Video node as: {os.path.basename(src)}')

    if dur > SEGMENT_SECONDS:
        n = math.ceil(dur / SEGMENT_SECONDS)
        print(f'\n⚠️  {dur:.1f}s exceeds SEGMENT_SECONDS={SEGMENT_SECONDS} → {n} segments')
        if SPLIT_IF_LONG:
            stem = os.path.splitext(os.path.basename(src))[0]
            outs = []
            for i in range(n):
                dst = f'{INPUT_DIR}/{stem}_seg{i:02d}.mp4'
                subprocess.run(['ffmpeg','-y','-loglevel','error','-ss',str(i*SEGMENT_SECONDS),
                                '-i',src,'-t',str(SEGMENT_SECONDS),'-c','copy',dst], check=False)
                outs.append(os.path.basename(dst))
            print('   ✅ wrote:', ', '.join(outs))
            print('   Render each, then concatenate the audio before utility C.')
        else:
            print('   Tick SPLIT_IF_LONG to cut it, or raise the threshold and test.')
"""))

# --- B
cells.append(md(r"""
### B. Prompt builder + vocabulary logbook

Two jobs. It rewrites emotional states into the acoustic events CLAP actually knows, and it keeps a running record in Drive of which phrasings worked — which ends up more useful than any published taxonomy, because it's calibrated to your prompts and your CFG.

**Emotional state → vocalisation.** `"man surprised"` has to be inferred as whatever a surprised person emits, which spreads across gasps, yelps, silence and laughter — you get mush, or the model ignores it. `"sharp gasp"` lands in a tight region.

Non-lexical vocalisations are all in distribution: screaming, laughing, gasping, coughing, grunting, crying. They sit in the same label taxonomy as door slams and dog barks. **The boundary is at words, not at human voices.**
"""))
cells.append(code(r"""
#@title B. Prompt builder + logbook { display-mode: "form" }
PHRASE  = ""          #@param {type:"string"}
VERDICT = "untested"  #@param ["untested", "good", "weak", "ignored"]
NOTE    = ""          #@param {type:"string"}
SHOW_LOG = True       #@param {type:"boolean"}

import os, json, datetime

EMOTION_MAP = {
    'surprised': 'a sharp gasp / a startled yelp',
    'scared':    'a short high-pitched scream',
    'in pain':   'a grunt and a sharp exhale',
    'amused':    'male laughter',
    'angry':     'a sharp shout, heavy footsteps',
    'sad':       'quiet sobbing, sniffling',
    'exhausted': 'heavy breathing, a long sigh',
}

LOG = f'{PROMPTS_DIR}/vocabulary.json'
log = json.load(open(LOG)) if os.path.exists(LOG) else []

if PHRASE.strip():
    log.append({'phrase': PHRASE.strip(), 'verdict': VERDICT, 'note': NOTE.strip(),
                'at': datetime.datetime.now().isoformat(timespec='seconds')})
    json.dump(log, open(LOG, 'w'), indent=2)
    print(f'✅ Logged: "{PHRASE.strip()}"  [{VERDICT}]')

    low = PHRASE.lower()
    hits = [(k, v) for k, v in EMOTION_MAP.items() if k in low]
    if hits:
        print('\n⚠️  Emotional state detected — CLAP wants the sound, not the feeling:')
        for k, v in hits:
            print(f'     "{k}"  →  "{v}"')
    if PHRASE.count(',') >= 3 and '.' not in PHRASE:
        print('\n💡 Reads like a tag list. Prose matches the training captions better —')
        print('   try a full sentence describing the soundtrack.')

if SHOW_LOG:
    print(f'\n--- vocabulary logbook ({len(log)} entries) ---')
    for v in ('good', 'weak', 'ignored', 'untested'):
        rows = [e for e in log if e['verdict'] == v]
        if rows:
            print(f'\n  [{v}]')
            for e in rows:
                note = f"   — {e['note']}" if e['note'] else ''
                print(f'    · {e["phrase"]}{note}')
    if not log:
        print('  (empty — probe phrases text-only in the UI, then log them here)')

print('\n--- emotion → vocalisation reference ---')
for k, v in EMOTION_MAP.items():
    print(f'  {k:<10} → {v}')
"""))

# --- C
cells.append(md(r"""
### C. Layered mix — loudness-matched

Takes the two passes from Step 8, normalises both to a common loudness, then mixes at your balance. **The normalisation is the point:** raw passes arrive at wildly different levels, and without matching them first the balance control does nothing useful.

`BALANCE` is the vocal layer's weight — `0.5` is even, lower buries the vocal under the physical layer.
"""))
cells.append(code(r"""
#@title C. Layered mix { display-mode: "form" }
PHYSICAL = "physical.flac"  #@param {type:"string"}
VOCAL    = "vocal.flac"     #@param {type:"string"}
BALANCE  = 0.4              #@param {type:"slider", min:0, max:1, step:0.05}
LUFS     = -23              #@param {type:"number"}
OUT_NAME = "mixed.wav"      #@param {type:"string"}

import os, subprocess

def resolve(p):
    return p if os.path.isabs(p) else f'{OUTPUT_DIR}/{p}'

a, b = resolve(PHYSICAL), resolve(VOCAL)
missing = [p for p in (a, b) if not os.path.exists(p)]
if missing:
    for p in missing:
        print(f'⚠️  Not found: {p}')
    have = [f for f in sorted(os.listdir(OUTPUT_DIR)) if f.lower().endswith(('.wav','.flac','.mp3'))]
    print('\n   In output_audio/:', ', '.join(have) if have else '(empty)')
else:
    out = resolve(OUT_NAME)
    wp, wv = 1.0 - BALANCE, BALANCE
    # loudnorm each input to the same LUFS target, then weighted sum.
    filt = (f'[0:a]loudnorm=I={LUFS}:TP=-1.5:LRA=11,volume={wp:.3f}[p];'
            f'[1:a]loudnorm=I={LUFS}:TP=-1.5:LRA=11,volume={wv:.3f}[v];'
            f'[p][v]amix=inputs=2:duration=longest:normalize=0[out]')
    r = subprocess.run(['ffmpeg','-y','-loglevel','error','-i',a,'-i',b,
                        '-filter_complex',filt,'-map','[out]',
                        '-ar','48000','-c:a','pcm_s16le',out],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print('❌ ffmpeg failed:\n', r.stderr[-1500:])
    else:
        print(f'✅ Mixed → {out}')
        print(f'   physical {wp:.2f} / vocal {wv:.2f}  @ {LUFS} LUFS')
        try:
            from IPython.display import Audio, display
            display(Audio(out))
        except Exception:
            pass
"""))

# --- D
cells.append(md(r"""
### D. Mux onto the source video

Attaches the finished track to your original clip without re-encoding the video. Output lands in Drive next to the audio.
"""))
cells.append(code(r"""
#@title D. Mux audio onto video { display-mode: "form" }
VIDEO = "my_clip.mp4"  #@param {type:"string"}
AUDIO = "mixed.wav"    #@param {type:"string"}
OUT   = ""             #@param {type:"string"}

import os, subprocess

v = VIDEO if os.path.isabs(VIDEO) else f'{INPUT_DIR}/{VIDEO}'
a = AUDIO if os.path.isabs(AUDIO) else f'{OUTPUT_DIR}/{AUDIO}'
o = (OUT if os.path.isabs(OUT) else f'{OUTPUT_DIR}/{OUT}') if OUT else \
    f'{OUTPUT_DIR}/{os.path.splitext(os.path.basename(v))[0]}_foley.mp4'

missing = [p for p in (v, a) if not os.path.exists(p)]
if missing:
    for p in missing:
        print(f'⚠️  Not found: {p}')
else:
    r = subprocess.run(['ffmpeg','-y','-loglevel','error','-i',v,'-i',a,
                        '-map','0:v:0','-map','1:a:0','-c:v','copy',
                        '-c:a','aac','-b:a','192k','-shortest',o],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print('❌ ffmpeg failed:\n', r.stderr[-1500:])
    else:
        print(f'✅ {o}  ({os.path.getsize(o)/1024**2:.1f} MB)')
"""))

# --- E
cells.append(md(r"""
### E. Vocabulary reference — VGGSound & AudioSet

**There is no published list of supported effects, and there isn't one to find.** The model is open-vocabulary: training captions were generated automatically with GenAU rather than drawn from a fixed taxonomy, and while the pipeline did apply categorical tags to balance the 100k-hour dataset, Tencent didn't publish that taxonomy.

The usable proxies are the label sets it's *evaluated* against. **VGGSound** (309 classes) is one of its eval sets, so those categories are a fair map of what's reliably in distribution. **AudioSet** (632 hierarchical classes) is the broader superset most audio classifiers derive from. Neither is a contract — but if a sound has a node in AudioSet, it almost certainly has representation in a 100k-hour web-video corpus.
"""))
cells.append(code(r"""
#@title E. Browse the label sets { display-mode: "form" }
SOURCE = "audioset"  #@param ["audioset", "vggsound"]
FILTER = ""          #@param {type:"string"}
LIMIT  = 60          #@param {type:"number"}

import json, urllib.request

URLS = {
    'audioset': 'https://raw.githubusercontent.com/audioset/ontology/master/ontology.json',
    'vggsound': 'https://raw.githubusercontent.com/hche11/VGGSound/master/data/vggsound.csv',
}

try:
    raw = urllib.request.urlopen(URLS[SOURCE], timeout=30).read().decode('utf-8', 'replace')
    if SOURCE == 'audioset':
        labels = sorted({n['name'] for n in json.loads(raw)})
    else:
        labels = sorted({ln.split(',')[2].strip().strip('"')
                         for ln in raw.splitlines() if ln.count(',') >= 3})
except Exception as e:
    labels = []
    print(f'⚠️  Could not fetch {SOURCE} ({e}).')
    print('   These are third-party URLs and may move; browse them on GitHub instead:')
    print('  ', URLS[SOURCE])

if labels:
    hits = [l for l in labels if FILTER.lower() in l.lower()] if FILTER else labels
    print(f'{SOURCE}: {len(labels)} labels, {len(hits)} matching "{FILTER}"\n')
    for l in hits[:int(LIMIT)]:
        print('  ·', l)
    if len(hits) > LIMIT:
        print(f'\n  … {len(hits) - int(LIMIT)} more — narrow with FILTER or raise LIMIT')
    print('\n💡 Found a candidate? Probe it text-only in the UI (no video input),')
    print('   then log the phrasing that worked with utility B.')
"""))

# ----------------------------------------------------------------- tips
cells.append(md(r"""
---
## 📋 Prompting & troubleshooting

### Prompting
- **Prose over tags.** GenAU-style captions. `A heavy body hits the water with a loud splash; water sloshes against the sides.` beats `splash, water, sloshing`.
- **Sounds, not events.** Describe the soundtrack, not the plot.
- **Vocalisations, not emotions.** `startled yelp`, not `surprised`. See utility **B**.
- **One layer per pass.** Physical and vocal separately, then mix with **C**.

### Troubleshooting
- **Audio drifts out of sync** → `frame_rate` doesn't match the clip. Run utility **A** and use the fps it prints. This is the most common failure by a wide margin.
- **Scream comes out weak or absent** → TV2A training pipelines filter speech-dominant clips to stop the model babbling, and screams sit right on that classifier's edge. If the visual event is obvious, visual conditioning may also be swamping the text. **Raise CFG to 6–7** and re-run; that's the first thing to try.
- **One element swamps the other** → that's what the two-pass workflow and utility **C** exist for.
- **Vocalisation doesn't track the mouth** → expected. Synchformer aligns to *visual motion onset*; it has no concept of mouth shape. For impacts the two usually coincide.
- **You need intelligible words** → out of scope. `voice_explorer.ipynb` / `wan22_s2v.ipynb` for TTS + lipsync, then mix the Foley underneath with **C**.
- **OOM** → set `PRECISION="fp8_e4m3fn"` in Step 5 *and* the loader node's quantization to `auto`/`fp8`, or add `--lowvram` to `EXTRA_ARGS` in Step 7.
- **`models/foley` is empty after a restart** → expected. Weights live on local disk by design; re-run Step 5 (~2 min).
- **First run is slow** → the Dependencies Loader is pulling SigLIP2 + CLAP (~3 GB). Cached for the rest of the session only.

### Lighter alternative
**MMAudio** is the smaller, faster V2A model and is often good enough for a single obvious impact. Foley quality is subjective and benchmark wins don't always survive contact with a specific clip, so it's worth an A/B — but confirm the current repo before wiring it in; it isn't included here.
"""))

nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "A100"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}
out = sys.argv[1]
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    f.write(chr(10))
print(f"wrote {out}: {len(cells)} cells")

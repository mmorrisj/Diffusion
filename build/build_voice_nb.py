# -*- coding: utf-8 -*-
"""Generate voice_explorer.ipynb — a lightweight Colab notebook (T4) for browsing,
auditioning and generating cloned voices with F5-TTS, feeding wan22_s2v.ipynb."""
import json
import sys

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": src.strip("\n")})


# --------------------------------------------------------------------------- #
md(r"""
# 🔊 Voice Explorer — find, audition and generate cloned voices (Colab T4)

The cheap half of the speech-to-video pipeline. No ComfyUI, no 16 GB models — three TTS engines on
a **T4**, so you can spend an hour trying voices for pennies and save the A100 for rendering in
`wan22_s2v.ipynb`:

| Engine | Best at | Section |
|---|---|---|
| **F5-TTS** | cleanest zero-shot clone of a reference clip | A–F |
| **Qwen3-TTS** | designing a voice from a *description*, per-line emotion (`instruct`), presets, 10 languages | J–M |
| **Dia** | non-verbal sounds — `(gasps)`, `(laughs)`, `(sighs)` | G–I |

Models are cached on Drive (`hf_cache/`, `models/qwen-tts/`) and shared with the S2V notebook's ComfyUI.

```
 LibriTTS-R (CC BY 4.0)  ──►  voice library with labels  ──►  browse & listen
                                                                    │
 your own recording  ────────────────────────────────────►  audition your line in N voices
                                                                    │
                                                          generate final lines ──►  Drive › input_audio/
                                                                                         │
                                                                                   wan22_s2v.ipynb
```

Everything is written to the same Drive folders the S2V notebook reads:

| Folder | Contents |
|---|---|
| `ComfyUI_Wan/voices/library/` | `libri_<id>.wav` + `.txt` reference clips, `catalog.csv` with labels |
| `ComfyUI_Wan/voices/` | your own reference clips (`my_voice.wav` + `.txt`) |
| `ComfyUI_Wan/input_audio/` | generated speech — `audition_*.wav` (tests) and your final lines |
| `ComfyUI_Wan/hf_cache/` | F5-TTS + Whisper + Dia checkpoints, cached once |
| `ComfyUI_Wan/models/qwen-tts/` | Qwen3-TTS checkpoints (~3.5 GB each), shared with the ComfyUI nodes |

**Runtime:** `Runtime > Change runtime type > T4 GPU`. CPU also works (auditions take ~30–60 s each instead of ~3 s).
""")

# --------------------------------------------------------------------------- #
md("## Step 1 — Check runtime")
code(r"""
import subprocess, torch
r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)
gpu = r.stdout.strip() or 'none'
print(f'GPU: {gpu}')
print('✅ GPU available — F5-TTS will be fast' if torch.cuda.is_available()
      else '⚠️  No GPU — everything still works, auditions just take ~30–60s each')
""")

# --------------------------------------------------------------------------- #
md(r"""
## Step 2 — Mount Google Drive

Same `ComfyUI_Wan` folder as the other notebooks. Model checkpoints are cached on Drive so the
1.5 GB F5-TTS/Whisper download only happens once.
""")
code(r"""
from google.colab import drive
import os

drive.mount('/content/drive')

DRIVE_BASE = '/content/drive/MyDrive/ComfyUI_Wan'
for d in ['voices', 'voices/library', 'input_audio', 'hf_cache']:
    os.makedirs(f'{DRIVE_BASE}/{d}', exist_ok=True)
    print(f'✅ {DRIVE_BASE}/{d}')

# Persist F5-TTS / Whisper / vocoder downloads across sessions.
os.environ['HF_HOME'] = f'{DRIVE_BASE}/hf_cache'
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
print(f'\n✅ Drive mounted — HF cache at {os.environ["HF_HOME"]}')
""")

# --------------------------------------------------------------------------- #
md(r"""
## Step 3 — Install F5-TTS, Qwen3-TTS, Dia + tools (~3 min)

Everything is installed here, **before any model is imported**, because all three engines must
share one `transformers` version: `qwen-tts` pins **4.57.3**, and Colab ships 5.x, whose
`check_model_inputs` decorator changed shape (→ `TypeError: check_model_inputs() missing 1
required positional argument: 'func'` if 5.x is the one in memory). F5-TTS and Dia both work on
4.57.3.

If the cell ends with **"restart the runtime"**, do `Runtime → Restart session`, then run Steps 2–4
again (installs are already on disk, so they're instant). This only happens on the first run.
""")
code(r"""
import subprocess, importlib.util, importlib.metadata, sys

TRANSFORMERS = '4.57.3'   # pinned by qwen-tts; F5-TTS and Dia are fine with it

# 1) transformers first, so nothing below pulls in a different one
if importlib.metadata.version('transformers') != TRANSFORMERS:
    !pip install -q "transformers=={TRANSFORMERS}"
    print(f'✅ transformers pinned to {TRANSFORMERS}')
else:
    print(f'✅ transformers {TRANSFORMERS} already installed')

# 2) the engines + helpers (skip what's present)
need = [p for p, m in [('f5-tts', 'f5_tts'), ('qwen-tts', 'qwen_tts'), ('datasets', 'datasets'),
                       ('soundfile', 'soundfile'), ('pandas', 'pandas')]
        if importlib.util.find_spec(m) is None]
if need:
    !pip install -q {' '.join(need)} "transformers=={TRANSFORMERS}"
    print(f'✅ Installed: {", ".join(need)}')
else:
    print('✅ Engines already installed')

# 3) system tools: ffmpeg (audio/video utils), sox (qwen-tts resampling backend)
for tool in ('ffmpeg', 'sox'):
    if subprocess.run(['which', tool], capture_output=True).returncode != 0:
        !apt-get install -y -q {tool} > /dev/null
    print(f'✅ {tool} ready')

# 4) if a different transformers was already imported into this kernel, a restart is required
loaded = sys.modules.get('transformers')
if loaded is not None and loaded.__version__ != TRANSFORMERS:
    print(f'\n⚠️  transformers {loaded.__version__} is already loaded in this kernel — '
          f'restart the runtime (Runtime → Restart session), then run Steps 2–4 again.')
else:
    print('\n✅ All installs complete')
""")

# --------------------------------------------------------------------------- #
md(r"""
## Step 4 — Load F5-TTS once

Defines `synth(ref_wav, ref_text, text, out_path)`. Used by the audition and generate cells below.
First run downloads the model (~1.3 GB) into the Drive cache. An empty `ref_text` makes F5-TTS
auto-transcribe the reference with Whisper (another one-time ~1.5 GB download).
""")
code(r"""
import os, subprocess

MODEL = 'F5TTS_v1_Base'

try:
    from f5_tts.api import F5TTS
    tts = F5TTS(model=MODEL, hf_cache_dir=os.environ.get('HF_HOME'))
    def synth(ref_wav, ref_text, text, out_path, seed=42, speed=1.0):
        tts.infer(ref_file=ref_wav, ref_text=ref_text, gen_text=text, file_wave=out_path,
                  seed=seed, speed=speed, show_info=lambda *a, **k: None)
    print(f'✅ F5-TTS loaded via API ({MODEL})')
except Exception as e:
    print(f'ℹ️  API load failed ({type(e).__name__}: {e}) — falling back to the CLI (reloads model per call)')
    def synth(ref_wav, ref_text, text, out_path, seed=42, speed=1.0):
        subprocess.run(['f5-tts_infer-cli', '--model', MODEL, '--ref_audio', ref_wav, '--ref_text', ref_text,
                        '--gen_text', text, '--speed', str(speed), '--output_dir', os.path.dirname(out_path),
                        '--output_file', os.path.basename(out_path)], check=True, capture_output=True)

def load_ref(voice_name):
    # voice_name like 'my_voice', 'f5_demo' or 'library/libri_3081' → (wav path, transcript or '')
    wav = f'{DRIVE_BASE}/voices/{voice_name}.wav'
    txt_path = f'{DRIVE_BASE}/voices/{voice_name}.txt'
    txt = open(txt_path).read().strip() if os.path.exists(txt_path) else ''
    if txt.startswith('TYPE THE EXACT'):
        txt = ''
    if not os.path.exists(wav):
        raise FileNotFoundError(wav)
    return wav, txt
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 🎙️ Voice library

### A. Build the LibriTTS-R library (run once, ~3 min)

One streaming pass over a split collects ~10 s of clean speech + its transcript for **every
speaker**, labelled with gender / pitch / speaking rate / expressiveness / accent from the
Parler-TTS annotations. Saved to `voices/library/` with a `catalog.csv`, so you never need to
run it again — add a split later and it just appends.

| `SPLITS` | Voices | Time |
|---|---|---|
| `['dev.clean']` | 40 | ~3 min |
| `['dev.clean', 'test.clean']` | ~79 | ~6 min |
| `['train.clean.100']` | ~250 | ~20 min |

These are audiobook narrators (LibriVox volunteers) — "read aloud" delivery, CC BY 4.0 (credit
the corpus if you publish).
""")
code(r"""
import io, os, csv, collections, time
import numpy as np, soundfile as sf
from datasets import load_dataset, Audio

SPLITS       = ['dev.clean']
TARGET_SEC   = 10               # seconds of speech per speaker (hard cap 15 — F5 cuts there)
MAX_SPEAKERS = None             # e.g. 12 to stop early while testing

LIB = f'{DRIVE_BASE}/voices/library'
os.makedirs(LIB, exist_ok=True)
SR = 24000
gap = np.zeros(int(0.3 * SR), dtype=np.float32)

# --- 1. speaker labels (tiny metadata dataset) ---
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

# --- 3. write wav + txt, merge into catalog.csv ---
cat_path = f'{LIB}/catalog.csv'
existing = {}
if os.path.exists(cat_path):
    with open(cat_path, newline='') as f:
        existing = {r['name']: r for r in csv.DictReader(f)}

for spk, c in clips.items():
    if not c['audio']:
        continue
    name = f'libri_{spk}'
    sf.write(f'{LIB}/{name}.wav', np.concatenate(c['audio']), SR)
    transcript = ' '.join(c['text'])
    open(f'{LIB}/{name}.txt', 'w').write(transcript)
    lab = labels.get(spk, {k: '?' for k in LABEL_KEYS})
    existing[name] = {'name': name, 'speaker_id': spk, **lab, 'seconds': round(c['sec'], 1), 'transcript': transcript}

rows = sorted(existing.values(), key=lambda r: r['name'])
with open(cat_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f'\n✅ {len(clips)} voices added — catalog now has {len(rows)} voices: {cat_path}')
""")

# --------------------------------------------------------------------------- #
md(r"""
### B. Browse the library

Keyword filters (substring match, case-insensitive) → table + a player per voice.
""")
code(r"""
import pandas as pd
from IPython.display import Audio, HTML, display

GENDER  = 'any'     # 'male' / 'female' / 'any'
PITCH   = ''        # low-pitch · slightly low-pitch · moderate pitch · slightly high-pitch · high-pitch · very high-pitch
RATE    = ''        # very slowly · slowly · slightly slowly · moderate speed · slightly fast · fast · very fast
STYLE   = ''        # monotone · slightly expressive and animated · expressive and animated · very expressive
ACCENT  = ''        # American · Canadian · ...
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

# --------------------------------------------------------------------------- #
md(r"""
### C. Audition — your line in several voices, side by side

The step that matters: a reference that *sounds* great can clone badly and vice versa, so compare
cloned output, not reference clips. Fixed seed → the only thing that changes between players is the voice.
""")
code(r"""
import os
from IPython.display import Audio, HTML, display

VOICES = ['library/libri_3081', 'library/libri_84', 'f5_demo']   # names under voices/
SCRIPT = "Hey, thanks for watching. Today I want to show you something I've been working on."
SEED   = 42

out_dir = f'{DRIVE_BASE}/input_audio'
for v in VOICES:
    try:
        ref, txt = load_ref(v)
    except FileNotFoundError as e:
        print(f'⚠️  missing {e}'); continue
    out = f'{out_dir}/audition_{os.path.basename(v)}.wav'
    synth(ref, txt, SCRIPT, out, seed=SEED)
    display(HTML(f'<b>{v}</b> → {os.path.basename(out)}'))
    display(Audio(out))

print('\nPick a winner and use it as VOICE in section E.')
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 🎤 Your own voice

### D. Prepare a reference clip from a recording

3–15 s of one person speaking clearly, nothing else under it. Upload the raw file to
`Drive › ComfyUI_Wan › voices/` (any format), then run. It trims, optionally strips music/noise with
Demucs, writes a clean 24 kHz mono `voices/<name>.wav`, and scaffolds `<name>.txt`.

Type the **exact** words spoken into the `.txt` for the best clone — or leave it blank and F5 will
auto-transcribe with Whisper (good enough for testing).
""")
code(r"""
import os, subprocess, shutil

RAW_CLIP   = f'{DRIVE_BASE}/voices/raw_voice.mp3'   # update
VOICE_NAME = 'my_voice'
START_SEC  = 0
CLIP_SEC   = 12                                      # ≤ 15
SEPARATE_VOCALS = True                               # False if it's already clean speech

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
    open(txt, 'w').write('')
    print(f'ℹ️  Created empty transcript (auto-transcribe). For a better clone, type the exact words into: {txt}')
else:
    print(f'✅ Transcript present: {txt}')

from IPython.display import Audio, display
display(Audio(final))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 🗣️ Generate the real lines

### E. Batch-generate lines for the S2V notebook

One voice, many lines. Each becomes `input_audio/<name>.wav`, ready for `LoadAudio` in the S2V
graph. Keep each line to a sentence or two — F5 is happiest there; for a paragraph, split it and
either render separate clips or join the WAVs (section F).
""")
code(r"""
import os, subprocess, math
from IPython.display import Audio, HTML, display

VOICE = 'library/libri_3081'      # or 'my_voice', 'f5_demo', ...
SPEED = 1.0                        # 0.8 slower · 1.2 faster
SEED  = 42
LINES = {
    'line_01': "Hey, thanks for watching. Today I want to show you something I've been working on.",
    'line_02': "It started as a small experiment, and honestly, I didn't expect it to go anywhere.",
}

ref, txt = load_ref(VOICE)
out_dir = f'{DRIVE_BASE}/input_audio'
FPS, CHUNK = 16, 77
for name, text in LINES.items():
    out = f'{out_dir}/{name}.wav'
    synth(ref, txt, text, out, seed=SEED, speed=SPEED)
    dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', out],
                               capture_output=True, text=True).stdout)
    chunks = math.ceil(dur * FPS / CHUNK)
    display(HTML(f'<b>{name}</b> — {dur:.1f}s → {math.ceil(dur*FPS)} frames, {chunks} chunk(s), '
                 f'{chunks-1} Extend node(s) &nbsp;<span style="color:#888">{text[:80]}</span>'))
    display(Audio(out))
print(f'\n✅ Saved to {out_dir} — load these with LoadAudio in wan22_s2v.ipynb')
""")

# --------------------------------------------------------------------------- #
md(r"""
### F. Join lines into one track (optional)

For a longer monologue: joins the WAVs in order with a short pause between them.
""")
code(r"""
import subprocess, os
PARTS = ['line_01', 'line_02']      # in order
OUT   = 'monologue_01'
PAUSE = 0.4                         # seconds between lines

out_dir = f'{DRIVE_BASE}/input_audio'
lst = '/tmp/join.txt'
with open(lst, 'w') as f:
    for p in PARTS:
        f.write(f"file '{out_dir}/{p}.wav'\n")
sil = '/tmp/sil.wav'
!ffmpeg -y -loglevel error -f lavfi -i anullsrc=r=24000:cl=mono -t {PAUSE} {sil}
with open(lst, 'w') as f:
    for i, p in enumerate(PARTS):
        if i: f.write(f"file '{sil}'\n")
        f.write(f"file '{out_dir}/{p}.wav'\n")
!ffmpeg -y -loglevel error -f concat -safe 0 -i {lst} -ar 24000 -ac 1 "{out_dir}/{OUT}.wav"
dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0',
                            f'{out_dir}/{OUT}.wav'], capture_output=True, text=True).stdout)
print(f'✅ {OUT}.wav — {dur:.1f}s')
from IPython.display import Audio, display
display(Audio(f'{out_dir}/{OUT}.wav'))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 🎁 Bundled demo voices (optional)

F5-TTS ships four example clips. This copies them into `voices/` as `f5_demo`, `f5_main`,
`f5_town`, `f5_country` so they show up alongside your own.
""")
code(r"""
import glob, os, subprocess, importlib.util

spec = importlib.util.find_spec('f5_tts')
ex = glob.glob(f'{list(spec.submodule_search_locations)[0]}/infer/examples')
if not ex:
    raise SystemExit('⚠️  F5-TTS examples not found — run Step 3 first')
ex = ex[0]
demos = {
    'f5_demo'    : (f'{ex}/basic/basic_ref_en.wav', 'Some call me nature, others call me mother nature.'),
    'f5_main'    : (f'{ex}/multi/main.flac',        ''),
    'f5_town'    : (f'{ex}/multi/town.flac',        ''),
    'f5_country' : (f'{ex}/multi/country.flac',     ''),
}
for name, (src, text) in demos.items():
    if not os.path.exists(src):
        print(f'⚠️  missing {src}'); continue
    dst = f'{DRIVE_BASE}/voices/{name}.wav'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', src, '-ac', '1', '-ar', '24000', dst], check=True)
    open(f'{DRIVE_BASE}/voices/{name}.txt', 'w').write(text)
    print(f'✅ {name}.wav  transcript: {"yes" if text else "auto"}')
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## ✍️ Qwen3-TTS — design a voice in prose, style per line, presets

Three things F5 can't do: **describe** a voice instead of finding a clip of it, give a per-line
**emotion/style instruction**, and speak **10 languages**. Runs here through the `qwen-tts`
package (1.7B bf16 ≈ 3.5 GB per variant; the T4 handles one at a time — the loader swaps them).
Same models the ComfyUI workflow uses, cached on Drive under `models/qwen-tts/`, so whichever
notebook downloads them first, the other reuses them.

| Mode | What you give it |
|---|---|
| **Design** | text + a casting-note description ("female, mid-30s, mezzo, breathy, slow, vocal fry at line ends") |
| **Custom** | text + a preset speaker (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden, Ono_Anna, Sohee) + optional `instruct` |
| **Clone** | text + any `voices/…` reference, like F5 |

Non-verbal sounds aren't a documented feature — keep Dia for `(gasps)`.

### J. Load Qwen3-TTS (once)
""")
code(r"""
import os, torch, transformers

# qwen-tts pins transformers 4.57.3; a different version in memory means Step 3 ran after
# something imported transformers — restart the runtime and re-run Steps 2–4.
if transformers.__version__ != '4.57.3':
    raise SystemExit(f'transformers {transformers.__version__} is loaded but qwen-tts needs 4.57.3 — '
                     'run Step 3, then Runtime → Restart session, then Steps 2–4 again.')
from qwen_tts import Qwen3TTSModel

import shutil, time
QWEN_STORE = f'{DRIVE_BASE}/models/qwen-tts/Qwen'    # durable copy on Drive, shared with the ComfyUI nodes
QWEN_LOCAL = '/content/models/qwen-tts/Qwen'         # what we actually load from (VM disk = fast)
os.makedirs(QWEN_STORE, exist_ok=True); os.makedirs(QWEN_LOCAL, exist_ok=True)
_qwen = {'name': None, 'model': None}

# A checkpoint is only usable if BOTH weight files are fully present (a config.json alone means a
# download was interrupted — huggingface leaves the big files as .incomplete under .cache).
WEIGHTS = ('model.safetensors', 'speech_tokenizer/model.safetensors')
def _complete(d):
    return all(os.path.exists(f'{d}/{w}') and os.path.getsize(f'{d}/{w}') > 100 * 1024**2 for w in WEIGHTS)

def _stage(name):
    # Ensure a complete copy in QWEN_LOCAL: from the Drive store if it has one, else from HF (then save to the store).
    local, store = f'{QWEN_LOCAL}/{name}', f'{QWEN_STORE}/{name}'
    if _complete(local):
        return local
    t0 = time.time()
    if _complete(store):
        print(f'📥 copying {name} from Drive to local disk…')
        shutil.copytree(store, local, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.cache'))
    else:
        try:
            import hf_transfer  # noqa: F401  (multi-threaded downloads)
        except ImportError:
            !pip install -q hf_transfer
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
        from huggingface_hub import snapshot_download
        print(f'⬇️  downloading {name} (~3.8 GB) to local disk…')
        snapshot_download(repo_id=f'Qwen/{name}', local_dir=local)
        if not _complete(local):
            raise SystemExit(f'{name} still incomplete after download — re-run this cell (downloads resume).')
        print(f'   ↗ saving a copy to Drive ({store}) for next session…')
        shutil.copytree(local, store, dirs_exist_ok=True, ignore=shutil.ignore_patterns('.cache'))
    print(f'✅ {name} ready in {time.time()-t0:.0f}s')
    return local

def qwen(variant):
    # variant: 'VoiceDesign' | 'Base' (clone) | 'CustomVoice'. Keeps one model resident; swaps on change.
    name = f'Qwen3-TTS-12Hz-1.7B-{variant}'
    if _qwen['name'] != name:
        local = _stage(name)
        _qwen['model'] = None; torch.cuda.empty_cache()
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        _qwen['model'] = Qwen3TTSModel.from_pretrained(local, device_map='cuda:0' if torch.cuda.is_available() else 'cpu', dtype=dtype)
        _qwen['name'] = name
        print(f'✅ loaded {name}')
    return _qwen['model']

def qwen_design(text, instruct, out_path, language='English'):
    import soundfile as sf
    wavs, sr = qwen('VoiceDesign').generate_voice_design(text=text, language=language, instruct=instruct)
    sf.write(out_path, wavs[0], sr); return out_path

def qwen_custom(text, speaker, out_path, instruct=None, language='English'):
    import soundfile as sf
    kw = {'instruct': instruct} if instruct else {}
    wavs, sr = qwen('CustomVoice').generate_custom_voice(text=text, language=language, speaker=speaker, **kw)
    sf.write(out_path, wavs[0], sr); return out_path

def qwen_clone(text, voice_name, out_path, language='English'):
    import soundfile as sf
    ref, txt = load_ref(voice_name)
    kw = {'ref_text': txt} if txt else {'x_vector_only_mode': True}
    wavs, sr = qwen('Base').generate_voice_clone(text=text, language=language, ref_audio=ref, **kw)
    sf.write(out_path, wavs[0], sr); return out_path

print('✅ helpers ready: qwen_design / qwen_custom / qwen_clone')
""")

md(r"""
### K. Design voices from descriptions — audition several at once

Write casting notes, not adjectives. Each description becomes `input_audio/design_<name>.wav`;
the ones you like, promote to references with the last cell so they can be cloned consistently.
""")
code(r"""
from IPython.display import Audio, HTML, display

SCRIPT = "Hey, thanks for watching. Today I want to show you something I've been working on."
DESIGNS = {
    'sultry_30s'  : "Female, mid-30s, mezzo-soprano, sultry and mature. Rich, velvety timbre, breathy and intimate, close to the mic. Relaxed larynx, slow pacing, subtle vocal fry at the ends of sentences. Confident, alluring.",
    'weathered_70s': "Elderly male, late 70s, deep weathered baritone with a dry, coarse texture. Slow, deliberate, rhythmic pacing with audible weary breath support. Stoic, humble, physically tired but undefeated.",
    'nervous_20s' : "Male, 25, office worker. Clear voice but hesitant — slight stammers, rising inflection, quick shallow breaths. Trying to sound composed and not quite managing it.",
}

out_dir = f'{DRIVE_BASE}/input_audio'
for name, desc in DESIGNS.items():
    out = f'{out_dir}/design_{name}.wav'
    qwen_design(SCRIPT, desc, out)
    display(HTML(f'<b>{name}</b><br><span style="color:#888">{desc[:120]}…</span>'))
    display(Audio(out))
""")

md(r"""
### L. Per-line style with presets (or a designed/cloned voice)

`instruct` is the emotion dial. Same text, same speaker, different instruction — listen to how far it moves.
""")
code(r"""
from IPython.display import Audio, HTML, display

SPEAKER = 'Ryan'      # Vivian · Serena · Uncle_Fu · Dylan · Eric · Ryan · Aiden · Ono_Anna · Sohee
TEXT    = "It's in the top drawer... wait, it's empty? No way, that's impossible! I'm sure I put it there!"
STYLES  = {
    'neutral'    : None,
    'panic'      : "Speak in an incredulous tone, with a hint of panic beginning to creep into your voice.",
    'deadpan'    : "Flat, bored, deadpan delivery, as if this happens every day.",
    'whisper'    : "Whisper urgently, as if someone might overhear.",
}

out_dir = f'{DRIVE_BASE}/input_audio'
for name, instr in STYLES.items():
    out = f'{out_dir}/style_{SPEAKER}_{name}.wav'
    qwen_custom(TEXT, SPEAKER, out, instruct=instr)
    display(HTML(f'<b>{SPEAKER} · {name}</b> <span style="color:#888">{instr or ""}</span>'))
    display(Audio(out))
""")

md(r"""
### M. Promote a designed voice to a reusable reference

A design is only reproducible with the same seed/description. Save a take you like as
`voices/<name>.wav` + `.txt` and from then on clone it (Qwen `qwen_clone`, or F5 in section E)
for line-to-line consistency.
""")
code(r"""
import shutil
TAKE      = f'{DRIVE_BASE}/input_audio/design_sultry_30s.wav'     # a file from section K or L
AS_VOICE  = 'sultry_30s'
SAID      = "Hey, thanks for watching. Today I want to show you something I've been working on."  # what that take says

shutil.copy(TAKE, f'{DRIVE_BASE}/voices/{AS_VOICE}.wav')
open(f'{DRIVE_BASE}/voices/{AS_VOICE}.txt', 'w').write(SAID)
print(f'✅ voices/{AS_VOICE}.wav — use VOICE = "{AS_VOICE}" in section E (F5) or qwen_clone(...)')

# quick check: clone it with Qwen and listen
from IPython.display import Audio, display
out = f'{DRIVE_BASE}/input_audio/clone_check_{AS_VOICE}.wav'
qwen_clone("And this is the same voice, cloned, saying something completely different.", AS_VOICE, out)
display(Audio(out))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 🎭 Dia — non-verbal sounds: (gasps), (sighs), (laughs)…

F5-TTS can't place a gasp on cue. **Dia** (Nari Labs, 1.6B, Apache 2.0) was trained on dialogue
with non-verbal markers, so you write them inline. Loaded through 🤗 Transformers (~4.4 GB in fp16,
fits on the T4 next to F5), voice-cloned from the same reference clips as F5.

**Supported tags:** `(laughs)` `(clears throat)` `(sighs)` `(gasps)` `(coughs)` `(singing)` `(sings)`
`(mumbles)` `(beep)` `(groans)` `(sniffs)` `(claps)` `(screams)` `(inhales)` `(exhales)` `(applause)`
`(burps)` `(humming)` `(sneezes)` `(chuckle)` `(whistles)`

**Rules of the road**
- Text must start with `[S1]`; a second speaker is `[S2]`, alternating. The cell adds `[S1]` for you.
- Aim for **5–20 s of output** per generation — shorter sounds unnatural, longer gets rushed.
- Cloning wants a **5–10 s** prompt *with its transcript* (the cell trims your reference and prepends the transcript).
- Output varies per generation. Fix `SEED` once you like a take. Tags "may produce unexpected results" — audition a few seeds.
- Cloning fidelity is below F5's. Typical split: **F5 for the clean lines, Dia for the beats that need a gasp or a laugh.**
- Output is 44.1 kHz — S2V/`LoadAudio` doesn't mind.

### G. Load Dia (once)
""")
code(r"""
import torch, transformers

# Dia landed in transformers 4.53 — the 4.57.3 pinned in Step 3 has it. Don't upgrade here:
# a newer transformers would break qwen-tts in the same kernel.
try:
    from transformers import DiaForConditionalGeneration, AutoProcessor
except ImportError:
    raise SystemExit(f'transformers {transformers.__version__} has no Dia classes — run Step 3 '
                     '(pins 4.57.3), then Runtime → Restart session, then Steps 2–4 again.')

DIA_ID = 'nari-labs/Dia-1.6B-0626'
dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
dia_processor = AutoProcessor.from_pretrained(DIA_ID)
dia_model = DiaForConditionalGeneration.from_pretrained(DIA_ID, torch_dtype=dtype, device_map='auto')
print(f'✅ Dia loaded on {dia_model.device} ({dtype})')

DIA_SR = 44100

def _load_prompt(voice_name, max_sec=10.0):
    # Reference clip → (44.1 kHz float array trimmed to max_sec, transcript). Transcribes with Whisper if no .txt.
    import soundfile as sf, torchaudio, numpy as np
    wav, txt = load_ref(voice_name)
    a, sr = sf.read(wav, dtype='float32')
    if a.ndim > 1:
        a = a.mean(axis=1)
    a = a[: int(max_sec * sr)]
    if sr != DIA_SR:
        a = torchaudio.functional.resample(torch.from_numpy(a), sr, DIA_SR).numpy()
    if not txt:
        from transformers import pipeline
        global _asr
        if '_asr' not in globals():
            _asr = pipeline('automatic-speech-recognition', model='openai/whisper-small',
                            device=0 if torch.cuda.is_available() else -1)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav') as f:
            sf.write(f.name, a, DIA_SR); txt = _asr(f.name)['text'].strip()
        print(f'ℹ️  auto-transcribed prompt: {txt}')
    return a, txt

def dia_synth(voice_name, text, out_path, seed=42, max_seconds=15, guidance_scale=3.0, temperature=1.8, top_p=0.90, top_k=45):
    # Voice-cloned generation. `text` may contain (gasps) etc. Returns out_path.
    audio, transcript = _load_prompt(voice_name)
    if not text.lstrip().startswith('[S'):
        text = '[S1] ' + text.strip()
    full = f'[S1] {transcript} {text}'
    inputs = dia_processor(text=[full], audio=audio, padding=True, return_tensors='pt').to(dia_model.device)
    prompt_len = dia_processor.get_audio_prompt_len(inputs['decoder_attention_mask'])
    torch.manual_seed(seed)
    with torch.inference_mode():
        out = dia_model.generate(**inputs, max_new_tokens=int(max_seconds * 128), guidance_scale=guidance_scale,
                                 temperature=temperature, top_p=top_p, top_k=top_k)
    decoded = dia_processor.batch_decode(out, audio_prompt_len=prompt_len)
    dia_processor.save_audio(decoded, out_path)
    return out_path
""")

md(r"""
### H. Generate a line with non-verbal tags
""")
code(r"""
import os, subprocess, math
from IPython.display import Audio, HTML, display

VOICE  = 'library/libri_3081'        # same names as everywhere else
SCRIPT = "(gasps) What was that? … No. No, no, no. (sighs) Okay. Okay, we're fine."
NAME   = 'dia_line_01'               # → input_audio/dia_line_01.wav
SEED   = 42
MAX_SECONDS = 15                     # generation budget; ~128 tokens per second of audio

out = f'{DRIVE_BASE}/input_audio/{NAME}.wav'
dia_synth(VOICE, SCRIPT, out, seed=SEED, max_seconds=MAX_SECONDS)
dur = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', out],
                           capture_output=True, text=True).stdout)
chunks = math.ceil(dur * 16 / 77)
display(HTML(f'<b>{NAME}</b> — {dur:.1f}s → {chunks} chunk(s), {chunks-1} Extend node(s)'))
display(Audio(out))
""")

md(r"""
### I. Tag audition — same line, several seeds

Tags render differently every generation. Listen to a handful, then pin the `SEED` you like in section H.
""")
code(r"""
import os
from IPython.display import Audio, HTML, display

VOICE  = 'library/libri_3081'
SCRIPT = "(gasps) Oh— you scared me. (laughs) Sorry. I didn't hear you come in."
SEEDS  = [1, 2, 3, 4]

for s in SEEDS:
    out = f'{DRIVE_BASE}/input_audio/audition_dia_seed{s}.wav'
    dia_synth(VOICE, SCRIPT, out, seed=s)
    display(HTML(f'<b>seed {s}</b>'))
    display(Audio(out))
""")

# --------------------------------------------------------------------------- #
md(r"""
---
## 📋 Next: render it

Open **`wan22_s2v.ipynb`** on an **A100**, run Steps 1–6, load the *Wan2.2 S2V* template and:

1. `LoadAudio` → the `input_audio/<line>.wav` you generated here (Utility E in that notebook copies it into `ComfyUI/input`)
2. `LoadImage` → your reference image
3. Add the number of *Extend* nodes section E printed for that line
4. Queue

### Tips
- **Audition before you commit.** 3 s per voice here vs. minutes per video there.
- **Transcript accuracy beats clip quality.** If a clone sounds "off", check the `.txt` first.
- **Punctuation controls pacing.** Commas and full stops in `LINES` give natural pauses; `SPEED` handles overall tempo.
- **Same seed, same voice, same text → same audio.** Change `SEED` to get a different take.
- **Non-English:** `F5TTS_v1_Base` is English-trained. Timbre transfers, pronunciation doesn't; look at Qwen3-TTS for other languages.

### Troubleshooting
- **`f5_tts.api` import error** → Step 4 falls back to the CLI automatically; slower but same output
- **Whisper download on every run** → `HF_HOME` not set; re-run Step 2 before Step 4
- **Clone sounds like a different person** → reference has music/noise; re-run D with `SEPARATE_VOCALS = True`
- **Words dropped or garbled** → line too long; split it
- **Library build is slow** → it's network-bound (streaming from Hugging Face), not GPU-bound; a faster runtime won't help
""")

# --------------------------------------------------------------------------- #
nb = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
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

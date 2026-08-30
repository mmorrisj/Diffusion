# Diffusion — ComfyUI on Colab notebooks

Colab notebooks for running Wan 2.2 / LTX-2 video generation and voice cloning with ComfyUI,
with models cached in Google Drive. All notebooks share one Drive layout:

| Drive folder | Purpose |
|---|---|
| `ComfyUI_Wan/models/…` | Wan 2.2 diffusion models, text encoder, VAE, LoRAs, wav2vec2, Qwen3-TTS |
| `ComfyUI_Wan/output/video/` | rendered videos (watched by the LastFrame tool on the PC) |
| `ComfyUI_Wan/voices/` · `input_audio/` | reference voice clips · generated speech |
| `ComfyUI_Wan/*.json` | ComfyUI workflows (copies in [`workflows/`](workflows/)) |
| `ComfyUI_LTX/models/…` | LTX-2 models (~41 GB, separate folder) |
| `ComfyUI_Qwen/last_frames/` | last-frame PNGs for image editing / chaining |

## Notebooks

| Notebook | Runtime | What it does |
|---|---|---|
| [`wan22_i2v_comfyui_colab.ipynb`](wan22_i2v_comfyui_colab.ipynb) | A100 | Wan 2.2 image-to-video (high/low-noise pair, lightning LoRAs). Configurable storage: Drive / GCS / custom / none, with per-session staging to local NVMe |
| [`wan22_i2v_comfyui_colab_hf.ipynb`](wan22_i2v_comfyui_colab_hf.ipynb) | A100 | I2V variant: weights pulled from Hugging Face each session, only LoRAs kept in Drive |
| [`wan22_s2v.ipynb`](wan22_s2v.ipynb) | A100 | Wan 2.2 **speech-to-video**: audio drives a talking video. Installs F5-TTS and Qwen3-TTS nodes so voices can be cloned/designed in-graph |
| [`voice_explorer.ipynb`](voice_explorer.ipynb) | T4 | Voice work without ComfyUI: browse a labelled LibriTTS-R voice library, audition lines, clone (F5-TTS), design voices from prose / per-line emotion (Qwen3-TTS), non-verbal tags like `(gasps)` (Dia) |
| [`ltx2_s2v.ipynb`](ltx2_s2v.ipynb) | A100-80GB / H100 | LTX-2 audio-driven I2V ("custom voice"): one-shot clips as long as the line, 720×1280 @ 24 fps |

Typical flow: pick/clone a voice and generate lines in `voice_explorer` (cheap), then render in
`wan22_s2v` or `ltx2_s2v` (expensive). Videos land in `ComfyUI_Wan/output/video/`.

## Workflows

- `workflows/Qwen3TTS_Voice_Generation_Collection_V3.json` — Qwen3-TTS voice design / clone / presets / multi-role dialogue (FaboroHacks)
- `workflows/ltx2_custom_voice.json` — LTX-2 custom-voice I2V (AI Verse), patched to the model filenames `ltx2_s2v.ipynb` downloads and to save into `output/video/`

## Regenerating notebooks

The three newer notebooks are generated from scripts in [`build/`](build/) so edits stay
consistent (shared launch cell, shared Drive layout):

```
python build/build_s2v_nb.py   wan22_s2v.ipynb
python build/build_voice_nb.py voice_explorer.ipynb
python build/build_ltx2_nb.py  ltx2_s2v.ipynb
```

The S2V/LTX-2 notebooks use the I2V notebook's launch cell: `TUNNEL="colab"` (same-origin Colab proxy, no
403 host/origin issues with ComfyUI ≥ 1.19), with `ngrok` and `cloudflare` as alternatives.

# Diffusion — ComfyUI on Colab notebooks

Colab notebooks for running Wan 2.2 / LTX-2 video generation, voice cloning and Foley
sound design with ComfyUI. All notebooks share one Drive layout:

| Drive folder | Purpose |
|---|---|
| `ComfyUI_Wan/models/…` | Wan 2.2 diffusion models, text encoder, VAE, LoRAs, wav2vec2, Qwen3-TTS |
| `ComfyUI_Wan/output/video/` | rendered videos (watched by the LastFrame tool on the PC) |
| `ComfyUI_Wan/voices/` · `input_audio/` | reference voice clips · generated speech |
| `ComfyUI_Wan/*.json` | ComfyUI workflows (copies in [`workflows/`](workflows/)) |
| `ComfyUI_LTX/models/…` | LTX-2 models (~41 GB, separate folder) |
| `ComfyUI_Qwen/last_frames/` | last-frame PNGs for image editing / chaining |
| `ComfyUI_Qwen/models/loras/` · `input_images/` · `output/` | Qwen-Image-Edit LoRAs · source images · edited images |
| `ComfyUI_Foley/input_videos/` · `output_audio/` · `prompts/` | clips to score · rendered Foley · prompt logbook |

## Notebooks

| Notebook | Runtime | What it does |
|---|---|---|
| [`wan22_i2v_comfyui_colab.ipynb`](wan22_i2v_comfyui_colab.ipynb) | A100 | Wan 2.2 image-to-video (high/low-noise pair, lightning LoRAs). Configurable storage: Drive / GCS / custom / none, with per-session staging to local NVMe |
| [`wan22_i2v_comfyui_colab_hf.ipynb`](wan22_i2v_comfyui_colab_hf.ipynb) | A100 | I2V variant: weights pulled from Hugging Face each session, only LoRAs kept in Drive |
| [`wan22_s2v.ipynb`](wan22_s2v.ipynb) | A100 | Wan 2.2 **speech-to-video**: audio drives a talking video. Installs F5-TTS and Qwen3-TTS nodes so voices can be cloned/designed in-graph |
| [`voice_explorer.ipynb`](voice_explorer.ipynb) | T4 | Voice work without ComfyUI: browse a labelled LibriTTS-R voice library, audition lines, clone (F5-TTS), design voices from prose / per-line emotion (Qwen3-TTS), non-verbal tags like `(gasps)` (Dia) |
| [`ltx2_s2v.ipynb`](ltx2_s2v.ipynb) | A100-80GB / H100 | LTX-2 audio-driven I2V ("custom voice"): one-shot clips as long as the line, 720×1280 @ 24 fps |
| [`qwen_image_edit_comfyui_colab_hf.ipynb`](qwen_image_edit_comfyui_colab_hf.ipynb) | A100 (L4 works) | Qwen-Image-Edit 2509 instruction-based image editing (up to 3 input images, 4-step Lightning LoRA). Weights (~28 GB) pulled from Hugging Face each session with Xet high-performance, parallel downloads; only LoRAs and images kept in Drive. Workflow JSON comes from [`mmorrisj/qwen_edit`](https://github.com/mmorrisj/qwen_edit) |
| [`hunyuan_foley_v2a.ipynb`](hunyuan_foley_v2a.ipynb) | L4 or better | HunyuanVideo-Foley **video-to-audio**: scores an existing clip with synced 48 kHz Foley and ambience. Runs the opposite direction from the rest — video + text in, audio out, muxed back onto the clip |

Typical flow: pick/clone a voice and generate lines in `voice_explorer` (cheap), then render in
`wan22_s2v` or `ltx2_s2v` (expensive). Videos land in `ComfyUI_Wan/output/video/`. To add Foley or
ambience to a finished clip, run it through `hunyuan_foley_v2a` — weights come from Hugging Face
each session, so it uses no Drive quota beyond your own media.

## Workflows

- `workflows/Qwen3TTS_Voice_Generation_Collection_V3.json` — Qwen3-TTS voice design / clone / presets / multi-role dialogue (FaboroHacks)
- `workflows/ltx2_custom_voice.json` — LTX-2 custom-voice I2V (AI Verse), patched to the model filenames `ltx2_s2v.ipynb` downloads and to save into `output/video/`

## Regenerating notebooks

The four newer notebooks are generated from scripts in [`build/`](build/) so edits stay
consistent (shared launch cell, shared Drive layout):

```
python build/build_s2v_nb.py   wan22_s2v.ipynb
python build/build_voice_nb.py voice_explorer.ipynb
python build/build_ltx2_nb.py  ltx2_s2v.ipynb
python build/build_foley_nb.py hunyuan_foley_v2a.ipynb
```

The S2V/LTX-2 notebooks use the I2V notebook's launch cell: `TUNNEL="colab"` (same-origin Colab proxy, no
403 host/origin issues with ComfyUI ≥ 1.19), with `ngrok` and `cloudflare` as alternatives.

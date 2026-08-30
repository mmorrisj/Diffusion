# -*- coding: utf-8 -*-
"""Patch the uploaded LTX-2 custom-voice workflow so its dropdown values match the files
the ltx2_s2v notebook downloads, outputs land in output/video/, and the example prompt/line
are neutral placeholders."""
import json, sys

src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src, encoding="utf-8"))
nodes = {n["id"]: n for n in d["nodes"]}
changes = []

def setw(nid, idx, value):
    old = nodes[nid]["widgets_values"][idx]
    nodes[nid]["widgets_values"][idx] = value
    changes.append(f"#{nid} {nodes[nid]['type']}[{idx}]: {old!r} -> {value!r}")

# Text encoder + embeddings connector (DualCLIPLoader)
setw(190, 0, "gemma_3_12B_it_fp8_scaled.safetensors")
setw(190, 1, "ltx-2-19b-embeddings_connector_distill_bf16.safetensors")

# Video outputs -> output/video/ so the LastFrame watcher picks them up; drop stale previews
for nid in (140, 359):
    w = nodes[nid]["widgets_values"]
    old = w.get("filename_prefix")
    w["filename_prefix"] = "video/LTX-2"
    w["videopreview"] = {"hidden": False, "paused": False, "params": {}}
    changes.append(f"#{nid} VHS_VideoCombine.filename_prefix: {old!r} -> 'video/LTX-2'")

# Neutral placeholders for the example prompt and spoken line
setw(121, 0, "A person speaks directly to camera in a softly lit room, natural expression, subtle head "
             "and shoulder movement, lips in sync with the audio. Static camera, shallow depth of field, "
             "cinematic, 4k.")
setw(372, 0, "Hey, thanks for watching. Today I want to show you something I've been working on.")

# Clear the creator's input filenames (user picks their own)
setw(167, 0, "")
setw(290, 0, "")

json.dump(d, open(dst, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print("\n".join(changes))
print(f"\nwrote {dst}")

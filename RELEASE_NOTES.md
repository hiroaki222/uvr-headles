# Release Notes

## v0.1.0 — Headless / AI-agent-native CLI, first cut (2026-05-18)

### In one line
The first release that runs Ultimate Vocal Remover **without a GUI** and lets
**AI agents drive the parameters interactively** through a CLI wrapper.

### Highlights
- 🎛️ **`uvr` command** — one-shot source separation (MDX-Net / VR / Demucs).
- 🤖 **Agent integration**: `uvr list-params` discovers the parameter space as
  JSON → pass a JSON config → `--json` collects results. A TTY interactive
  mode is also included.
- 🪶 **Lighter**: all GUI-only deps (pyglet / tkinterdnd2 / matchering /
  cryptography / opencv …) removed. Only the separation runtime remains.
- 🧩 **Unmodified reuse**: the core separation logic (`separate.py` / `lib_v5`
  / `demucs`) is upstream as-is. Only `ModelData` was ported to be GUI-free.
- 📦 Installed via **`pip install -e .`**. Not on PyPI; clone + build.
- ⚖️ MIT (inherited from upstream). The GUI body is preserved under `legacy/`.

### What works in this release
- ✅ End-to-end MDX-Net separation verified (`UVR-MDX-NET Inst HQ 3`, CPU →
  Vocals + Instrumental stems + `--json` summary).
- ✅ `list-models`, `list-params`, package import, friendly error guidance
  when dependencies are missing.
- ⏳ VR / Demucs share the same code path but are not yet e2e-verified.

### Getting started
```bash
git clone https://github.com/hiroaki222/uvr-headles.git && cd uvr-headles
python -m venv .venv && source .venv/bin/activate
pip install -e .
uvr list-models
```
Fetch model weights separately into `models/` (see README).

### Acknowledgements
The separation engine and trained models are the work of Anjok07 / aufr33 and
the UVR core developers. This fork only adds a headless CLI layer on top.

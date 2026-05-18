# Changelog

Change history specific to this fork. Follows
[Keep a Changelog](https://keepachangelog.com/) and
[Semantic Versioning](https://semver.org/). The upstream
(Anjok07/ultimatevocalremovergui) history is out of scope.

## [Unreleased]

## [0.1.0] - 2026-05-18

Initial GUI-removal + CLI / AI-agent interface release.

### Added
- `uvr_headless` package and the `uvr` CLI (console_scripts in `pyproject.toml`).
- Subcommands: `list-models` / `list-params` / `separate` / `interactive`.
- AI-agent discover/run interface:
  `list-params` (emit the parameter space as JSON) → `--config` (specify via
  JSON) → `--json` (structured results). Fully local.
- `uvr_headless/core/model_data.py`: `ModelData` physically ported from UVR.py
  in a GUI-free form (path constants / `model_hash_table` /
  `load_model_hash_data` redefined; `BASE_PATH` resolves to the repo root and
  is overridable via env `UVR_BASE_PATH`).
- `uvr_headless/headless_root.py`: `HeadlessRoot` standing in for the GUI
  `root` (MainWindow) — lazily materializes `*_var` via `__getattr__`, tolerant
  of unknown vars.
- Restored `LICENSE` (MIT, inheriting the upstream copyright notice and stating
  the fork).
- `RELEASE_NOTES.md` and this `CHANGELOG.md`.

### Changed
- README fully rewritten around the fork / CLI.
- `requirements.txt` reduced to the separation runtime minimum (GUI-only deps
  removed: pyglet, tkinterdnd2, matchering, cryptography, opencv-python,
  Pillow, pyperclip, wget, screeninfo, samplerate, kthread, altgraph,
  playsound, pyrubberband, soundstretch, …).
- `get_model_data_from_popup` stubbed for headless (messagebox / `root.pop_up_*`
  removed; unknown models surface an explicit error via `model_status=False`).
- `yaml` / `ml_collections` made lazy imports (`ModelData` can be constructed
  without them for non-MDX-C models).

### Fixed
- Removed `Dora==0.0.3` from dependencies. It is unused in the inference path
  (demucs' `from dora.log import fatal` is commented out) and pulls the
  deprecated `sklearn` package, which makes `pip install` fail. `diffq` is
  still required and kept.
- Replaced the stale platform-split soundfile pin
  (`SoundFile==0.11.0` / `PySoundFile==0.9.0.post1; darwin`) with
  `soundfile>=0.12.1`. The old darwin pin ships no bundled `libsndfile` and
  fails at runtime without a system install; modern `soundfile` bundles
  `libsndfile` wheels (incl. macOS arm64).

### Moved
- `UVR.py` → `legacy/UVR.py` (GUI body kept for reference, unmodified).
- old `requirements.txt` → `legacy/requirements-gui.txt` (deps for the GUI).

### Notes / Known limitations
- End-to-end separation is verified for MDX-Net (`UVR-MDX-NET Inst HQ 3`, CPU).
  VR and Demucs share the same path but are not yet e2e-verified in this fork.
- Vendored `demucs/apply.py` / `demucs/utils.py` contain `import tkinter` from
  an upstream modification; left unmodified to stay mergeable with upstream.
- Secondary models / ensemble / vocal split are disabled in the MVP.

[Unreleased]: https://github.com/hiroaki222/uvr-headles/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hiroaki222/uvr-headles/releases/tag/v0.1.0

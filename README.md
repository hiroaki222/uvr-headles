# uvr-headless

**A GUI-less, AI-agent-native CLI wrapper** — a fork of [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) (Anjok07 / aufr33).

It reuses UVR's source-separation engine (MDX-Net / VR / Demucs) as-is, strips the tkinter GUI, and exposes only a **command line** plus a **JSON-based discover/run interface**. It is designed primarily so that AI agents (e.g. Claude Code) can *discover parameters → set them → run* without a human in the loop.

> This is a fork. The separation models, core algorithms, and model-author credits belong to the upstream UVR core developers (Anjok07, aufr33). This fork only adds: GUI removal + CLI/agent interface + packaging.

---

## What changed vs. upstream

| | upstream UVR | uvr-headless |
|---|---|---|
| Entry point | tkinter GUI | CLI (`uvr` command) |
| Parameter control | GUI widgets | CLI flags / JSON config / interactive |
| Dependencies | full GUI stack (pyglet, tkinterdnd2, matchering, cryptography, opencv …) | separation runtime only |
| GUI body | `UVR.py` | moved to `legacy/UVR.py` (kept for reference) |
| Agent integration | none | `list-params` (JSON) → `--config` (JSON) → `--json` (results) |

Only `ModelData` was physically ported into `uvr_headless/core/model_data.py` in a GUI-free form. `separate.py` / `lib_v5` / `demucs` / `gui_data/constants.py` are reused unmodified.

---

## Installation

Not published to PyPI. The intended workflow is **clone + editable install**.

```bash
git clone https://github.com/hiroaki222/uvr-headles.git
cd uvr-headles
python -m venv .venv && source .venv/bin/activate
pip install -e .          # installs the minimal deps from pyproject.toml + the `uvr` command
# or: pip install -r requirements.txt
```

`list-models` / `list-params` work even without torch etc. (structure discovery). `separate` needs the deps above plus model weights.

### Getting model weights

Model weights (`.onnx` / `.pth` …) are **not bundled**. Browse and fetch them
from the CLI, exactly like the GUI's model list — no manual URLs:

```bash
uvr list-models                 # browse every downloadable model + status
uvr download "Kim Vocal 2"      # fetch its weight file(s) to the right place
```

`download` resolves the name against the upstream catalog
(`gui_data/model_manual_download.json`), downloads to the correct
`models/...` subdirectory, and skips files that already exist (`--force` to
redownload). `list-models` shows `[x]` for downloaded, `[ ]` otherwise.

---

## Usage

### Browse & download models

```bash
uvr list-models            # human-readable (GUI-parity catalog + status)
uvr list-models --json     # machine-readable
uvr download "Kim Vocal 2" # download a model's weight file(s)
```

### Discover parameters (for agents)

```bash
uvr list-params            # emit every tunable parameter as a JSON schema
```

Example output (excerpt):

```json
{
  "params": [
    {"name": "mdx_segment_size", "type": "int", "default": 256, "choices": null,
     "help": "MDX segment size. Larger = higher quality / more memory"}
  ],
  "required": ["input", "output", "model"],
  "methods": ["mdx", "vr", "demucs"]
}
```

### Run separation

With flags:

```bash
uvr separate -i song.wav -o out/ -m "UVR-MDX-NET Inst HQ 3" \
    --primary_only true --mdx_segment_size 256
```

With a JSON config (recommended for agents):

```bash
echo '{"mdx_segment_size": 512, "denoise": "Standard", "gpu": true}' > params.json
uvr separate -i song.wav -o out/ -m "Kim Vocal 2" --config params.json --json
```

Batch a directory and collect results as JSON:

```bash
uvr separate -i ./tracks/ -o ./stems/ -m "UVR-MDX-NET Inst HQ 3" --json
```

### Interactive mode

```bash
uvr interactive            # answer one prompt at a time to fill in parameters
```

### Agent integration flow

```
uvr list-models  ->  agent sees every model + which are downloaded
uvr download "<name>"  ->  agent fetches the chosen model on demand
uvr list-params  ->  agent learns the parameter space
uvr separate --config <generated.json> --json  ->  run & collect structured results
```

So "make sample.mp3 an acapella" becomes, for an agent:
`list-models` → `download "Kim Vocal 2"` → `separate -i sample.mp3 -o out/ -m "Kim Vocal 2"`.

Everything runs locally. No network or external service required.

---

## Known limitations

- **e2e verified for MDX-Net only**: end-to-end separation has been verified
  with MDX-Net (`UVR-MDX-NET Inst HQ 3`, CPU, producing Vocals + Instrumental
  stems and the `--json` summary). VR and Demucs go through the same code path
  and `ModelData` supports them, but they have not yet been e2e-verified in
  this fork.
- **Vendored demucs imports tkinter**: `demucs/apply.py` / `demucs/utils.py`
  contain `import tkinter` from an upstream modification (pulled in via
  `separate`). It is part of the standard library so it is fine on typical
  setups, but a fully headless environment without tkinter needs it for
  `separate`. The vendored code is left unmodified to stay mergeable with
  upstream.
- **Secondary models / ensemble / vocal split are disabled**: the MVP only does
  single-model separation. `ModelData` supports them but the headless helpers
  are stubbed out.

---

## Credits

- Separation engine & trained models: **Anjok07**, **aufr33** and the UVR core
  developers ([Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui))
- Demucs v3/v4: original authors (Facebook Research / Demucs authors)
- This fork (GUI removal + CLI/agent interface): hiroaki222

## License

MIT, inherited from upstream UVR's MIT license. See [LICENSE](LICENSE); change
history in [CHANGELOG.md](CHANGELOG.md) / [RELEASE_NOTES.md](RELEASE_NOTES.md).

#!/usr/bin/env python3
"""
uvr-headless CLI
================

Run Ultimate Vocal Remover (UVR) without a GUI. Designed so AI agents can
*discover parameters -> set them -> run*: it offers a JSON-based discover/run
interface plus a TTY interactive mode.

Subcommands:
    list-models   list available models     (light deps, works immediately)
    list-params   tunable parameters as JSON (light deps, works immediately)
    separate      run separation            (needs torch/onnxruntime + weights)
    interactive   fill parameters one prompt at a time

list-models / list-params work without torch etc.
separate / interactive need the runtime deps plus model weights.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# core.model_data is light (no torch/numpy/yaml). It is also the source of
# truth for the path constants.
from uvr_headless.core import model_data
from uvr_headless.headless_root import HeadlessRoot, default_for
from gui_data.constants import (
    DEF_OPT,
    DEMUCS_ARCH_TYPE,
    DENOISE_M,
    DENOISE_NONE,
    DENOISE_S,
    FLAC,
    MDX23_CONFIG_CHECKS,
    MDX_ARCH_TYPE,
    MP3,
    MP3_BIT_RATES,
    NORMAL_REPO,
    VR_ARCH_TYPE,
    WAV,
    WAV_TYPE,
)

MDX_MODELS_DIR = model_data.MDX_MODELS_DIR
VR_MODELS_DIR = model_data.VR_MODELS_DIR
DEMUCS_MODELS_DIR = model_data.DEMUCS_MODELS_DIR
DEMUCS_NEWER_REPO_DIR = model_data.DEMUCS_NEWER_REPO_DIR
MDX_C_CONFIG_PATH = model_data.MDX_C_CONFIG_PATH
MODEL_DL_CATALOG_JSON = os.path.join(
    model_data.BASE_PATH, "gui_data", "model_manual_download.json"
)
MDX_NAME_SELECT_JSON = model_data.MDX_MODEL_NAME_SELECT
DEMUCS_NAME_SELECT_JSON = model_data.DEMUCS_MODEL_NAME_SELECT

METHOD_ALIASES = {"mdx": MDX_ARCH_TYPE, "vr": VR_ARCH_TYPE, "demucs": DEMUCS_ARCH_TYPE}

# ---- Parameters exposed to agents / the CLI ----
# var: the root.<var>_var key ModelData reads / type: value type / choices
PARAM_SPEC = [
    {"name": "mdx_segment_size", "var": "mdx_segment_size", "type": "int",
     "choices": None, "help": "MDX segment size. Larger = higher quality / more memory (e.g. 256)"},
    {"name": "overlap_mdx23", "var": "overlap_mdx23", "type": "str",
     "choices": None, "help": "MDX23(C) overlap (default '8')"},
    {"name": "overlap_mdx", "var": "overlap_mdx", "type": "str",
     "choices": [DEF_OPT, "0.25", "0.5", "0.75", "0.99"],
     "help": "Legacy MDX overlap ('Default' or 0.0-0.99)"},
    {"name": "mdx_batch_size", "var": "mdx_batch_size", "type": "str",
     "choices": None, "help": "Inference batch size ('Default'=1 or an int string)"},
    {"name": "denoise", "var": "denoise_option", "type": "str",
     "choices": [DENOISE_NONE, DENOISE_S, DENOISE_M], "help": "Denoise mode"},
    {"name": "compensate", "var": "compensate", "type": "str",
     "choices": None, "help": "Volume compensation ('Auto' or a float string)"},
    {"name": "semitone_shift", "var": "semitone_shift", "type": "str",
     "choices": None, "help": "Pitch shift in semitones (0 disables)"},
    {"name": "match_freq_pitch", "var": "is_match_frequency_pitch", "type": "bool",
     "choices": [True, False], "help": "Whether to do frequency/pitch matching"},
    {"name": "invert_spec", "var": "is_invert_spec", "type": "bool",
     "choices": [True, False], "help": "Derive the secondary stem via spectral inversion"},
    {"name": "gpu", "var": "is_gpu_conversion", "type": "bool",
     "choices": [True, False], "help": "Use GPU (CUDA/MPS)"},
    {"name": "normalization", "var": "is_normalization", "type": "bool",
     "choices": [True, False], "help": "Normalize the output"},
    {"name": "primary_only", "var": "is_primary_stem_only", "type": "bool",
     "choices": [True, False], "help": "Save the primary stem only"},
    {"name": "secondary_only", "var": "is_secondary_stem_only", "type": "bool",
     "choices": [True, False], "help": "Save the secondary stem only"},
    {"name": "wav_type", "var": "wav_type_set", "type": "str",
     "choices": list(WAV_TYPE), "help": "WAV bit format"},
    {"name": "save_format", "var": "save_format", "type": "str",
     "choices": [WAV, FLAC, MP3], "help": "Output format"},
    {"name": "mp3_bitrate", "var": "mp3_bit_set", "type": "str",
     "choices": list(MP3_BIT_RATES), "help": "MP3 bitrate (when save_format=MP3)"},
    {"name": "device_set", "var": "device_set", "type": "str",
     "choices": None, "help": "GPU device selector ('Default' or an index)"},
]
PARAM_BY_NAME = {p["name"]: p for p in PARAM_SPEC}


# ---------------------------------------------------------------------------
# list-models / list-params : light deps (no torch)
# ---------------------------------------------------------------------------
def _load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _download_catalog():
    """Build the downloadable-model catalog from gui_data/model_manual_download.json.

    Mirrors the model list the UVR GUI offers. Each entry::

        {"name": <name to pass to `separate -m` / `download`>,
         "arch": "mdx"|"vr"|"demucs",
         "downloaded": bool,
         "files": [{"fname", "url", "dest"}]}

    Light-only (json + path constants); no torch.
    """
    d = _load_json(MODEL_DL_CATALOG_JSON)
    mdx_map = _load_json(MDX_NAME_SELECT_JSON)  # onnx/ckpt stem -> friendly name
    catalog = []

    def _entry(name, arch, files):
        downloaded = all(
            os.path.isfile(f["dest"]) and os.path.getsize(f["dest"]) > 0
            for f in files
        )
        catalog.append({"name": name, "arch": arch,
                        "downloaded": downloaded, "files": files})

    # MDX-Net: value is a single ".onnx" filename, hosted on NORMAL_REPO.
    for key, fname in d.get("mdx_download_list", {}).items():
        stem = os.path.splitext(fname)[0]
        name = mdx_map.get(stem, mdx_map.get(fname, stem))
        _entry(name, "mdx", [{
            "fname": fname, "url": NORMAL_REPO + fname,
            "dest": os.path.join(MDX_MODELS_DIR, fname)}])

    # VR: single ".pth" filename on NORMAL_REPO; -m name is the stem.
    for key, fname in d.get("vr_download_list", {}).items():
        _entry(os.path.splitext(fname)[0], "vr", [{
            "fname": fname, "url": NORMAL_REPO + fname,
            "dest": os.path.join(VR_MODELS_DIR, fname)}])

    # MDX23 (MDX-C): {ckpt_filename: yaml_config_name}. ckpt on NORMAL_REPO,
    # yaml config on MDX23_CONFIG_CHECKS.
    for key, pair in d.get("mdx23_download_list", {}).items():
        files = []
        for ckpt, yml in pair.items():
            files.append({"fname": ckpt, "url": NORMAL_REPO + ckpt,
                          "dest": os.path.join(MDX_MODELS_DIR, ckpt)})
            files.append({"fname": yml, "url": MDX23_CONFIG_CHECKS + yml,
                          "dest": os.path.join(MDX_C_CONFIG_PATH, yml)})
            name = mdx_map.get(ckpt, os.path.splitext(ckpt)[0])
        _entry(name, "mdx", files)

    # Demucs: {filename: full_url}. v3/v4 go to the newer repo dir.
    for key, fmap in d.get("demucs_download_list", {}).items():
        prefix, _, disp = key.partition(": ")
        newer = ("v3" in prefix) or ("v4" in prefix)
        dest_dir = DEMUCS_NEWER_REPO_DIR if newer else DEMUCS_MODELS_DIR
        files = [{"fname": fn, "url": url,
                  "dest": os.path.join(dest_dir, fn)}
                 for fn, url in fmap.items()]
        _entry(disp, "demucs", files)

    return catalog


def cmd_list_models(args):
    catalog = _download_catalog()
    if args.json:
        print(json.dumps(catalog, indent=2, ensure_ascii=False))
        return 0
    for arch in ("mdx", "vr", "demucs"):
        rows = [c for c in catalog if c["arch"] == arch]
        print(f"\n[{arch.upper()}]  ({sum(c['downloaded'] for c in rows)}/{len(rows)} downloaded)")
        for c in rows:
            mark = "[x]" if c["downloaded"] else "[ ]"
            print(f"  {mark} {c['name']}")
    print('\nDownload one with:  uvr download "<name>"')
    print('Then run:           uvr separate -i IN -o OUT -m "<name>"')
    return 0


def _http_download(url, dest, label=""):
    """Stream a URL to dest with a simple progress line. Validates it is not
    an HTML error page. Returns the number of bytes written."""
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    req = urllib.request.Request(url, headers={"User-Agent": "uvr-headless"})
    with urllib.request.urlopen(req) as resp:
        ctype = resp.headers.get("Content-Type", "")
        total = int(resp.headers.get("Content-Length", 0) or 0)
        if "text/html" in ctype:
            raise SystemExit(f"Got an HTML page, not a file (bad URL?): {url}")
        done = 0
        last = -1  # last reported step (percent, or MB when size unknown)
        is_tty = sys.stdout.isatty()
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                step = int(done * 100 / total) if total else (done >> 20)
                if step == last:
                    continue  # throttle: only on a new step (agent-friendly)
                last = step
                if total:
                    msg = f"  {label} {done >> 20}/{total >> 20}MB ~{step:3d}%"
                else:
                    msg = f"  {label} {step}MB"
                sys.stdout.write((f"\r{msg}" if is_tty else msg + "\n"))
                sys.stdout.flush()
    if is_tty:
        sys.stdout.write("\n")
    if os.path.getsize(tmp) == 0:
        os.remove(tmp)
        raise SystemExit(f"Downloaded 0 bytes: {url}")
    os.replace(tmp, dest)
    return os.path.getsize(dest)


def cmd_download(args):
    catalog = _download_catalog()
    q = args.model
    exact = [c for c in catalog if c["name"] == q]
    ci = [c for c in catalog if c["name"].lower() == q.lower()]
    sub = [c for c in catalog if q.lower() in c["name"].lower()]
    matches = exact or ci or sub
    if not matches:
        raise SystemExit(
            f"No model matches '{q}'. See `uvr list-models`.")
    if len(matches) > 1 and not (exact or ci):
        names = "\n  ".join(f'{c["name"]}  [{c["arch"]}]' for c in matches[:20])
        raise SystemExit(
            f"'{q}' is ambiguous, candidates:\n  {names}\n"
            f"Re-run with the exact name.")
    entry = matches[0]
    print(f'Model: {entry["name"]}  [{entry["arch"]}]  '
          f'({len(entry["files"])} file(s))')
    got = []
    for fobj in entry["files"]:
        if (os.path.isfile(fobj["dest"]) and os.path.getsize(fobj["dest"]) > 0
                and not args.force):
            print(f'  skip (exists): {fobj["fname"]}')
            got.append(fobj["dest"])
            continue
        size = _http_download(fobj["url"], fobj["dest"], label=fobj["fname"])
        print(f'  saved: {fobj["dest"]} ({size >> 20}MB)')
        got.append(fobj["dest"])
    result = {"name": entry["name"], "arch": entry["arch"], "files": got}
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f'Done. Use:  uvr separate -i IN -o OUT '
              f'--method {entry["arch"]} -m "{entry["name"]}"')
    return 0


def cmd_list_params(args):
    spec = [{
        "name": p["name"], "type": p["type"],
        "default": default_for(p["var"]),
        "choices": p["choices"], "help": p["help"],
    } for p in PARAM_SPEC]
    payload = {
        "params": spec,
        "required": ["input", "output", "model"],
        "methods": list(METHOD_ALIASES.keys()),
        "usage": {
            "cli": "uvr separate -i IN -o OUT -m MODEL [--<param> VALUE ...]",
            "json": "uvr separate -i IN -o OUT -m MODEL --config params.json",
            "config_schema": '{ "<param-name>": <value>, ... }  (the "name" keys from list-params)',
        },
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


# ---------------------------------------------------------------------------
# separate : heavy deps (torch/onnxruntime + model weights)
# ---------------------------------------------------------------------------
def _build_overrides(param_values):
    ov = {}
    for name, val in param_values.items():
        if name not in PARAM_BY_NAME:
            raise SystemExit(f"Unknown parameter: {name} (check `uvr list-params`)")
        ov[PARAM_BY_NAME[name]["var"]] = val
    return ov


def _coerce(spec, raw):
    if spec["type"] == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).strip().lower() in ("1", "true", "yes", "on", "y")
    if spec["type"] == "int":
        return int(raw)
    return str(raw)


def cmd_separate(args, param_values=None):
    in_path = os.path.abspath(args.input)
    inputs = []
    exts = (".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg")
    if os.path.isdir(in_path):
        inputs = [os.path.join(in_path, f) for f in sorted(os.listdir(in_path))
                  if f.lower().endswith(exts)]
    elif os.path.isfile(in_path):
        inputs = [in_path]
    else:
        raise SystemExit(f"Input not found: {in_path}")
    if not inputs:
        raise SystemExit(f"No audio files in input directory: {in_path}")

    export_path = os.path.abspath(args.output)
    os.makedirs(export_path, exist_ok=True)

    method = METHOD_ALIASES.get(args.method)
    if method is None:
        raise SystemExit(f"Unknown method: {args.method} (mdx/vr/demucs)")

    pv = {}
    if args.config:
        pv.update(_load_json(os.path.abspath(args.config)))
    if param_values:
        pv.update(param_values)
    for p in PARAM_SPEC:
        cli_val = getattr(args, p["name"], None)
        if cli_val is not None:
            pv[p["name"]] = _coerce(p, cli_val)
    for k in list(pv.keys()):
        if k in PARAM_BY_NAME:
            pv[k] = _coerce(PARAM_BY_NAME[k], pv[k])
    overrides = _build_overrides(pv)
    overrides["chosen_process_method"] = method

    # --- Heavy import (torch etc. required from here) ---
    try:
        from separate import (
            SeperateDemucs,
            SeperateMDX,
            SeperateMDXC,
            SeperateVR,
        )
    except ImportError as e:
        raise SystemExit(
            f"Missing dependency: {e}\n"
            f"Run `pip install -e .` (or `pip install -r requirements.txt`) first."
        )

    mappers = {
        "vr_hash_MAPPER": _load_json(model_data.VR_HASH_JSON),
        "mdx_hash_MAPPER": _load_json(model_data.MDX_HASH_JSON),
        "mdx_name_select_MAPPER": _load_json(model_data.MDX_MODEL_NAME_SELECT),
        "demucs_name_select_MAPPER": _load_json(model_data.DEMUCS_MODEL_NAME_SELECT),
    }
    model_data.root = HeadlessRoot(overrides, mappers)

    model = model_data.ModelData(args.model, selected_process_method=method)
    if not model.model_status:
        raise SystemExit(
            f"Could not resolve model: '{args.model}'\n"
            f"  - check the weight file exists under models/\n"
            f"  - check model_data.json has a hash mapping for it\n"
            f"Use `uvr list-models` to see candidates."
        )

    def write_to_console(text, base_text=""):
        sys.stdout.write(text)
        sys.stdout.flush()

    def set_progress_bar(step, inference_iterations=0):
        pct = int(min(1.0, step + inference_iterations) * 100)
        sys.stdout.write(f"\r  progress ~{pct:3d}%")
        sys.stdout.flush()

    results = []
    for idx, audio_file in enumerate(inputs, start=1):
        base = os.path.splitext(os.path.basename(audio_file))[0]
        audio_file_base = f"{idx}_{base}"
        print(f"\n[{idx}/{len(inputs)}] {os.path.basename(audio_file)} -> separating...")

        process_data = {
            "model_data": model,
            "export_path": export_path,
            "audio_file_base": audio_file_base,
            "audio_file": audio_file,
            "set_progress_bar": set_progress_bar,
            "write_to_console": write_to_console,
            "process_iteration": model_data.root.process_iteration,
            "cached_source_callback": model_data.root.cached_source_callback,
            "cached_model_source_holder": model_data.root.cached_model_source_holder,
            "list_all_models": [],
            "is_ensemble_master": False,
            "is_4_stem_ensemble": False,
        }

        if method == VR_ARCH_TYPE:
            sep = SeperateVR(model, process_data)
        elif method == MDX_ARCH_TYPE:
            sep = SeperateMDXC(model, process_data) if model.is_mdx_c \
                else SeperateMDX(model, process_data)
        else:
            sep = SeperateDemucs(model, process_data)

        sep.seperate()

        produced = sorted(
            os.path.join(export_path, f) for f in os.listdir(export_path)
            if f.startswith(audio_file_base)
        )
        results.append({"input": audio_file, "outputs": produced})
        print(f"\n  done -> {len(produced)} file(s)")

    summary = {"method": args.method, "model": args.model,
               "params": pv, "results": results}
    if args.json:
        print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def cmd_interactive(args):
    print("=== uvr-headless interactive ===")
    args.input = input("input file/directory: ").strip()
    args.output = input("output directory: ").strip()
    args.method = input("method [mdx]: ").strip() or "mdx"
    args.model = input("model name (see list-models): ").strip()
    args.config = None
    args.json = False

    pv = {}
    print("\nParameters (Enter to keep the default):")
    for p in PARAM_SPEC:
        d = default_for(p["var"])
        ch = f" {p['choices']}" if p["choices"] else ""
        raw = input(f"  {p['name']} (default {d!r}){ch}: ").strip()
        if raw != "":
            pv[p["name"]] = _coerce(p, raw)
    for p in PARAM_SPEC:
        setattr(args, p["name"], None)
    return cmd_separate(args, param_values=pv)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="uvr",
        description="Run UVR without a GUI / JSON interface for AI agents",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_lm = sub.add_parser("list-models", help="list available models")
    p_lm.add_argument("--json", action="store_true", help="JSON output")
    p_lm.set_defaults(func=cmd_list_models)

    p_lp = sub.add_parser("list-params", help="emit tunable parameters as JSON")
    p_lp.set_defaults(func=cmd_list_params)

    p_dl = sub.add_parser("download", help="download a model's weight files")
    p_dl.add_argument("model", help='model name (see list-models)')
    p_dl.add_argument("--force", action="store_true",
                      help="re-download even if files exist")
    p_dl.add_argument("--json", action="store_true", help="JSON output")
    p_dl.set_defaults(func=cmd_download)

    p_sep = sub.add_parser("separate", help="run separation")
    p_sep.add_argument("-i", "--input", required=True, help="input file or directory")
    p_sep.add_argument("-o", "--output", required=True, help="output directory")
    p_sep.add_argument("-m", "--model", required=True, help="model name")
    p_sep.add_argument("--method", default="mdx", help="mdx|vr|demucs (default mdx)")
    p_sep.add_argument("--config", default=None, help="parameter JSON file")
    p_sep.add_argument("--json", action="store_true", help="emit results as JSON")
    for p in PARAM_SPEC:
        p_sep.add_argument(f"--{p['name']}", default=None, help=p["help"])
    p_sep.set_defaults(func=cmd_separate)

    p_int = sub.add_parser("interactive", help="interactive mode")
    p_int.set_defaults(func=cmd_interactive)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

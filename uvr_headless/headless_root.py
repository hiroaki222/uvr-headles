"""Headless root: a stand-in for the GUI (tkinter) ``root`` (MainWindow).

UVR's ``ModelData`` / ``separate.py`` normally read ``root.<name>_var.get()``
from the GUI. Injecting :class:`HeadlessRoot` into
``uvr_headless.core.model_data.root`` lets the existing logic run as-is without
ever starting a GUI.
"""

from __future__ import annotations

from gui_data.constants import ALL_STEMS, DEFAULT, DEFAULT_DATA, MDX_ARCH_TYPE, NO_MODEL, WAV

# Defaults for vars that ModelData reads but that are not in DEFAULT_DATA.
# (In the GUI these are defined separately in MainWindow.__init__.)
EXTRA_DEFAULTS = {
    "device_set": DEFAULT,
    "is_deverb_vocals": False,
    "deverb_vocal_opt": "Main Vocals Only",
    "is_use_opencl": False,
    "is_primary_stem_only": False,
    "is_secondary_stem_only": False,
    "is_save_inst_set_vocal_splitter": False,
    "set_vocal_splitter": NO_MODEL,
    "is_set_vocal_splitter": False,
    "mdxnet_stems": ALL_STEMS,
    "chosen_process_method": MDX_ARCH_TYPE,
    "ensemble_main_stem": "",
    "wav_type_set": "PCM_16",
    "mp3_bit_set": "320k",
    "save_format": WAV,
}


def default_for(var_key):
    """Effective default for a var key (DEFAULT_DATA first, then EXTRA)."""
    if var_key in DEFAULT_DATA:
        return DEFAULT_DATA[var_key]
    if var_key in EXTRA_DEFAULTS:
        return EXTRA_DEFAULTS[var_key]
    return ""


class _FakeVar:
    """Minimal tk.*Var stand-in. get()/set() only."""

    __slots__ = ("_v",)

    def __init__(self, value):
        self._v = value

    def get(self):
        return self._v

    def set(self, value):
        self._v = value


class HeadlessRoot:
    """GUI stand-in injected into ``model_data.root``.

    ``__getattr__`` lazily materializes ``<name>_var`` accesses as
    :class:`_FakeVar`, resolving the initial value in the order
    overrides → DEFAULT_DATA → EXTRA_DEFAULTS. Unknown var names do not raise,
    so it stays robust as new models are added upstream.
    """

    def __init__(self, overrides, mappers):
        object.__setattr__(self, "_store", {})
        object.__setattr__(self, "_overrides", dict(overrides or {}))
        for k, v in mappers.items():
            object.__setattr__(self, k, v)

    def __getattr__(self, name):
        if name.startswith("_") or not name.endswith("_var"):
            raise AttributeError(name)
        store = object.__getattribute__(self, "_store")
        if name in store:
            return store[name]
        key = name[:-4]  # "xxx_var" -> "xxx"
        overrides = object.__getattribute__(self, "_overrides")
        val = overrides[key] if key in overrides else default_for(key)
        var = _FakeVar(val)
        store[name] = var
        return var

    # Derived plain attribute: root.wav_type_set (mirrors UVR.py:6141-6144)
    @property
    def wav_type_set(self):
        wt = self.wav_type_set_var.get()
        if wt == "32-bit Float":
            return "FLOAT"
        if wt == "64-bit Float":
            return "FLOAT" if self.save_format_var.get() != WAV else "DOUBLE"
        return wt

    # --- Helpers ModelData calls (MVP: single model; secondary/split off) ---
    def return_ensemble_stems(self, is_primary=False):
        ens = self.ensemble_main_stem_var.get().partition("/")
        return ens[0] if is_primary else (ens[0], ens[2])

    def check_only_selection_stem(self, checktype):
        return False

    def process_determine_secondary_model(self, *a, **k):
        return None, None

    def process_determine_demucs_pre_proc_model(self, *a, **k):
        return None

    def process_determine_vocal_split_model(self, *a, **k):
        return None

    # --- Cache callbacks used by process_data ---
    def cached_source_callback(self, process_method, model_name=None):
        return None, None

    def cached_model_source_holder(self, process_method, sources, model_name=None):
        return None

    def process_iteration(self):
        return None

# legacy/

The upstream (Anjok07/ultimatevocalremovergui) **GUI version**, kept here for
reference. uvr-headless does not use it.

- `UVR.py` — the tkinter GUI body (7000+ lines). The CLI only ports `ModelData`
  out of it into `uvr_headless/core/model_data.py`.
- `requirements-gui.txt` — dependencies needed to run the GUI version (includes
  pyglet / tkinterdnd2 / matchering / cryptography / playsound, …).

Only if you want to run the GUI:

```bash
pip install -r legacy/requirements-gui.txt
python legacy/UVR.py
```

For normal headless CLI usage this directory is not needed.

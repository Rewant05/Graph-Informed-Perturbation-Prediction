# Installation

Clone the repository or extract the release ZIP. Run commands from the repository root. Use a suitable Python/PyTorch/CUDA environment; requirements are compatibility ranges, not a portable lockfile. The environment snapshot records the author's supplied working setup.

```bash
python -m pip install -r requirements.txt
```

GEARS training additionally requires its backend and compatible PyG:

```bash
python -m pip install -r requirements-gears.txt
```

For evidence verification alone, `python verify_release.py` needs only the Python standard library. Saved plotting needs NumPy, SciPy and Matplotlib.

Optional exploratory Harmony uses `requirements-preprocessing.txt`. Do not upgrade a working CUDA environment merely to build documentation.

Documentation can be built without importing/training the models:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs work/docs-html
```

Open `work/docs-html/index.html` in a browser. Hosted Read the Docs activation is explained in the hosting guide.

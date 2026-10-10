"""Resolve preserved source modules from this checkout."""
import importlib
import sys
from pathlib import Path

def load(name):
    source = Path(__file__).resolve().parents[1] / 'code'
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module(name)
    if Path(module.__file__).resolve().parent != source.resolve():
        raise ImportError(f'{name} resolved outside this repository')
    return module

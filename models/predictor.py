"""STRING-assisted interaction-transfer predictor: exact preserved algorithm.

learned_predictions fits nearest-residual and kernel-ridge branches to training
residuals. The runner adds these to measured additive single-gene effects.
This predictor is not a neural network, VAE or diffusion architecture.
"""
from ._legacy import load

__all__ = ['learned_predictions','pair_key','project_edges']

def __getattr__(name):
    if name in __all__:return getattr(load('gears_pair_graph_branch_matched'),name)
    raise AttributeError(name)

def main():
    import argparse, subprocess, sys
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gears-results')
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    args=[sys.executable,str(root/'run_study.py'),'predictor']
    if a.gears_results:args += ['--gears-results',str(Path(a.gears_results).resolve())]
    return subprocess.run(args,cwd=root,check=True).returncode

if __name__ == '__main__':
    raise SystemExit(main())

"""STRING-conditioned latent diffusion: public names and training entry point.

Definitions remain in the preserved source files. PairDDPM/PairConditioner are
in run_validation_selected.py; LatentVAE/GraphDDPM in the prototype module.
The complete training protocol is run_study.py diffusion, including SVD fitting,
validation selection, preprocessing/checkpoint saving and matched evaluation.
"""
from ._legacy import load

__all__ = ['LatentVAE','GraphDDPM','PairConditioner','PairDDPM','train_pair_variant','run_fold']

def __getattr__(name):
    if name in ('LatentVAE','GraphDDPM'):
        return getattr(load('compare_dual_graph_ot_vae_ddpm'),name)
    if name in ('PairConditioner','PairDDPM','train_pair_variant','run_fold'):
        return getattr(load('run_validation_selected'),name)
    raise AttributeError(name)

def main():
    import argparse, subprocess, sys
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fold',type=int,choices=range(1,6),default=1)
    p.add_argument('--all-folds',action='store_true')
    p.add_argument('--controls',action='store_true')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--gears-results')
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    args=[sys.executable,str(root/'run_study.py'),'diffusion-controls' if a.controls else 'diffusion',
          '--device',a.device]
    args += ['--all-folds'] if a.all_folds else ['--fold',str(a.fold)]
    if a.gears_results: args += ['--gears-results',str(Path(a.gears_results).resolve())]
    return subprocess.run(args,cwd=root,check=True).returncode

if __name__ == '__main__':
    raise SystemExit(main())

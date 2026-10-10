# Verification scope

## Checked in this package review

- Python source syntax and local import closure.
- Five disjoint train/validation/test pair folds; 101 unique test pairs overall.
- Baseline and saved GEARS alignment across model reports.
- Uploaded baseline, predictor, diffusion and diffusion-control rerun results versus references: maximum per-pair score difference 0.0.
- Original GEARS checkpoint rescoring: maximum per-pair difference 2.14e-9.
- Expression export alignment with saved diffusion errors.
- Figure generation from supplied saved evidence, release-file hashes and relative documentation links.
- Core model and GEARS source files preserved byte-for-byte from the uploaded package.

These checks read the author's supplied reports. They do not independently repeat GPU training. The supplied original checkpoint rescoring was performed on the author's machine. Fresh GEARS fold 1 differs and full fresh five-fold training is not established.

## Dataset verification

The official archive and working processed file match the recorded checksums. Raw `A549` annotations are preserved; byte provenance is not a biological annotation audit. `check_dataset.py` verifies shape/axis metadata rather than resolving that discrepancy.

## What `verify_release.py` does

It uses only the Python standard library to check source syntax, local imports, split partitions, stored numerical evidence, uploaded reproduction comparisons and the manifest. It does not install dependencies, fetch data, train a model or prove that a new environment will reproduce training trajectories.

Run meaningful fresh experiments using `docs/REPRODUCTION.md`, retaining their actual outputs even when they differ from reference scores.

## Version 0.2 interfaces and preprocessing

The Sphinx HTML site builds with warnings treated as errors. CLI help for the model/preprocessing interfaces was checked. The predictor public API resolves to the original algorithm object, and all 23 numerical source files remain byte-identical to the prior release. New preprocessing tests cover rejecting invalid counts, preserving input/counts/annotations, exploratory expression preservation, batch metadata validation, optional Harmony coordinates, and H5AD output/overwrite protection. These are synthetic software tests, not a new biological benchmark. No diffusion GPU training was performed during this packaging change.

Run preprocessing tests with `python -m unittest discover -s tests -v` after installing the main analysis dependencies. The optional Harmony test skips if harmonypy is absent.

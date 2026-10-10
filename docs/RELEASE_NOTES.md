# Repository review and cleanup

- Reworked the README with overview, model diagram, results, quick start, figure gallery, references and navigable documentation.
- Preserved the uploaded numerical model/GEARS code unchanged.
- Included both architectures, additive/nearest baselines, five-fold GEARS wrappers, frozen splits/graphs, results and plotting code.
- Included the supplied UMAP and all result figures; PNG/PDF pairs are intentional.
- Retained uploaded reproduction reports and computed pair-level audit summaries.
- Updated dataset checksums, annotation caveat and GEARS checkpoint-versus-fresh-training distinction.
- Normalised the environment snapshot to UTF-8.
- Excluded local `work/` checkpoints, caches and generated working outputs from this source package. Nothing was deleted from the author's Windows folders. Original weights/data must be retained locally.
- No GitHub push, publication claim or license assignment was performed.

## Version 0.2: model navigation and documentation

- Added public `models.diffusion` and `models.predictor` interfaces to exact original class/function objects and runners; numerical code remains unchanged.
- Added explicit benchmark/raw/exploratory preprocessing routes and optional Harmony. Existing benchmark data and scores remain unchanged.
- Preserved the historical Norman preprocessing script; corrected the new total-count/log1p layer naming.
- Added Sphinx documentation and Read the Docs configuration; hosted service activation is still required.
- Added Git attributes to preserve release-hashed bytes across platforms.

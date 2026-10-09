# Data, graphs and environment

## Exact expression input

All current matched comparisons use the same processed Norman file: 91,205 cells, 5,045 genes and 284 condition labels. Do not substitute `norman_log1p_10k_v2.h5ad`, which uses a different gene axis/scale. No expression dataset is bundled.

- Official GEARS Norman download: https://dataverse.harvard.edu/api/access/datafile/6154020
- Archive size: 168,758,985 bytes
- Archive MD5: `cdc41d6050e619c37fd9dd44d440e2b4`
- Archive SHA256: `c1938353f6f41829c9137f073ecb843ce4d4114528b001b48312b019f739d877`
- Extracted `norman/perturb_processed.h5ad` SHA256: `23ffb0fac6a847ff927cf7509d80d85052bfefbfb97610786a2dafaaefa0b6a0`

The downloaded archive matched the official endpoint by MD5, and the working H5AD matched the archived H5AD by SHA256. This establishes byte provenance and does not independently validate every biological annotation.

The distributed file labels every cell `A549`, despite the expected Norman K562 context. Preserve these raw annotations. This is not evidence that GEARS and the study models used different datasets. The UMAP uses neutral cell labels rather than relabelling cells. Record this discrepancy in any biological dataset description.

Run `check_dataset.py --data PATH` for shape, conditions and gene-axis agreement; separately compare the file checksum. The uploaded fingerprint is in `evidence/reproduction/dataset_fingerprint.json`.

## Frozen graphs

STRING: version-12 API, human 9606, functional associations, score threshold 400, supplied balanced1 mapping. These associations are not all direct physical interactions. The matched pipeline filters the frozen mapping to supported nodes.

Hi-C: supplied GRCh38 mapping, promoter flank 2,000 bp, strict promoter–promoter contacts only. Distal nearest-TSS hypothesis edges are not used as strict contacts or claimed causal. In the matched generative graph: 125 nodes, 170 STRING edges, 14 Hi-C edges; STRING connects 50 of 65 targets and Hi-C connects 9.

Raw Hi-C rebuilding is incomplete in this source release: the historical v2 script in `archive/` needs `prepare_k562_hic_gene_graph.py` and the original loop/GTF files. The main study runs with the included frozen mapping. No raw-map regeneration claim is made.

## Environment

`environment-used.txt` is the uploaded working-environment snapshot, normalised to UTF-8. The supplied environment audit reports Python 3.12.10, PyTorch 2.6.0+cu124, CUDA 12.4, PyG 2.8.0.post1, NumPy 2.5.2, Scanpy 1.12.3 and AnnData 0.13.2. GEARS was imported from site-packages; its distribution metadata was unavailable in that audit. The wrapper requirements identify cell-gears 0.1.2, but this is not proof of an exact original backend commit.

`requirements.txt` contains compatibility ranges, not a lockfile. GPU/CUDA wheels and Windows packages may not install unchanged on another platform. Preserve your working environment while auditing, and capture a new freeze for each new run. The original GEARS training environment was not fully captured, and fresh training is not established as numerically identical.

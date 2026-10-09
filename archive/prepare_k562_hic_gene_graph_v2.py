r"""Audit two K562 Hi-C mapping rules without reading perturbation outcomes.

PowerShell:
  python .\prepare_k562_hic_gene_graph_v2.py .\norman_log1p_10k_v2.h5ad .\string_balanced1.json .\GSE237898_ENCFF256ZMD_loops_GRCh38.bedpe.gz .\gencode.v50.annotation.gtf.gz --out .\k562_hic_mapping_v2.json

The strict graph connects measured genes whose PROMOTERS overlap opposite loop
anchors. Unlike v1, it considers every measured gene as a possible neighbor,
not only the 230 nodes selected for STRING. These are physical contacts, not
regulatory or causal directions.

The separate proximity HYPOTHESIS assigns a distal anchor to its nearest
measured-gene TSS within 50 kb, only if the opposite anchor maps to a promoter
in the original STRING node set. A nearest gene is not a validated enhancer
target. Never mix these hypothetical edges with the strict graph silently.
"""

import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

from prepare_k562_hic_gene_graph import (GENE_NAME, PRIMARY_CHROMS,
                                         chromosome, read_text)


VERSION = "2026-09-29-k562-hic-gene-mapping-v2"


def all_measured_tss(gtf_path, measured):
    canonical = {g.upper(): g for g in measured}
    if len(canonical) != len(measured):
        raise ValueError("Duplicate measured gene symbols, ignoring case")
    loci = defaultdict(set)
    with read_text(gtf_path) as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "gene" or fields[6] not in ("+", "-"):
                continue
            chrom = chromosome(fields[0])
            if chrom not in PRIMARY_CHROMS:
                continue
            match = GENE_NAME.search(fields[8])
            if match is None or match.group(1).upper() not in canonical:
                continue
            start, end = int(fields[3]), int(fields[4])
            if start < 1 or end < start:
                raise ValueError("Invalid GTF gene coordinates")
            gene = canonical[match.group(1).upper()]
            loci[gene].add((chrom, (start if fields[6] == "+" else end) - 1))
    ambiguous = sorted(g for g, positions in loci.items() if len(positions) != 1)
    by_chrom = defaultdict(list)
    for gene, positions in loci.items():
        if len(positions) == 1:
            chrom, tss = next(iter(positions))
            by_chrom[chrom].append((tss, gene))
    index = {}
    for chrom, entries in by_chrom.items():
        entries.sort()
        index[chrom] = ([position for position, _ in entries], entries)
    return index, ambiguous, sorted(set(measured) - set(loci))


def promoters_at(index, chrom, start, end, flank):
    positions, entries = index.get(chrom, ([], []))
    return {gene for _, gene in entries[bisect_left(positions, start - flank):
                                      bisect_left(positions, end + flank)]}


def nearest_tss(index, chrom, start, end, limit):
    """Assign only a unique nearest measured gene; ties remain unmapped."""
    positions, entries = index.get(chrom, ([], []))
    if not positions:
        return None
    mid = (start + end - 1) / 2
    pivot = bisect_left(positions, mid)
    # If the interval spans TSSs, consider all entries in that interval too.
    lo, hi = bisect_left(positions, start), bisect_left(positions, end)
    candidates = entries[max(0, lo - 1):min(len(entries), hi + 1)]
    if not candidates:
        candidates = entries[max(0, pivot - 1):min(len(entries), pivot + 1)]
    distances = [(max(start - pos, pos - (end - 1), 0), name)
                 for pos, name in candidates]
    best = min(distance for distance, _ in distances)
    closest = {name for distance, name in distances if distance == best}
    return next(iter(closest)) if best <= limit and len(closest) == 1 else None


def map_bedpe(loop_path, index, core_nodes, flank=2000, nearest_bp=50000):
    strict = defaultdict(int)
    hypothetical = defaultdict(int)
    stats = {"loops_read": 0, "loops_on_primary_chromosomes": 0,
             "loops_with_any_measured_promoter": 0,
             "loops_with_two_measured_promoters": 0,
             "loops_with_core_promoter": 0,
             "loops_with_distal_proximity_hypothesis": 0}
    with read_text(loop_path) as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 6:
                raise ValueError("BEDPE needs at least six columns")
            if fields[1].lower() in ("start1", "chromstart1"):
                continue
            c1, c2 = chromosome(fields[0]), chromosome(fields[3])
            s1, e1, s2, e2 = (int(fields[i]) for i in (1, 2, 4, 5))
            if min(s1, s2) < 0 or e1 <= s1 or e2 <= s2:
                raise ValueError("Invalid BEDPE coordinates")
            stats["loops_read"] += 1
            if c1 not in PRIMARY_CHROMS or c2 not in PRIMARY_CHROMS:
                continue
            stats["loops_on_primary_chromosomes"] += 1
            left = promoters_at(index, c1, s1, e1, flank)
            right = promoters_at(index, c2, s2, e2, flank)
            stats["loops_with_any_measured_promoter"] += bool(left or right)
            stats["loops_with_two_measured_promoters"] += bool(left and right)
            stats["loops_with_core_promoter"] += bool((left | right) & core_nodes)
            for a in left:
                for b in right:
                    if a != b:
                        strict[tuple(sorted((a, b)))] += 1
            # Distal endpoint assignments are explicitly a separate hypothesis.
            proposed = set()
            if left & core_nodes and not right:
                nearest = nearest_tss(index, c2, s2, e2, nearest_bp)
                if nearest is not None:
                    proposed |= {tuple(sorted((a, nearest))) for a in left & core_nodes
                                 if a != nearest}
            if right & core_nodes and not left:
                nearest = nearest_tss(index, c1, s1, e1, nearest_bp)
                if nearest is not None:
                    proposed |= {tuple(sorted((b, nearest))) for b in right & core_nodes
                                 if b != nearest}
            stats["loops_with_distal_proximity_hypothesis"] += bool(proposed)
            for pair in proposed:
                hypothetical[pair] += 1
    return strict, hypothetical, stats


def graph_summary(support, targets, core):
    connected = {gene for pair in support for gene in pair}
    return {"edges": [[a, b, n] for (a, b), n in sorted(support.items())],
            "connected_measured_genes": len(connected),
            "connected_string_nodes": len(connected & core),
            "connected_perturbation_targets": len(connected & targets)}


def build_graph(measured, manifest, loop_path, gtf_path, flank=2000,
                nearest_bp=50000):
    if manifest.get("gene_sha256") != hashlib.sha256(
            "\n".join(measured).encode()).hexdigest():
        raise ValueError("STRING manifest and expression gene order differ")
    core = set(manifest["nodes"])
    if len(core) != len(manifest["nodes"]) or not core <= set(measured):
        raise ValueError("STRING nodes must be unique measured genes")
    targets = set(manifest["train_targets"] + manifest["heldout"])
    index, ambiguous, missing = all_measured_tss(gtf_path, measured)
    strict, hypothetical, stats = map_bedpe(
        loop_path, index, core, flank, nearest_bp)
    return {
        "version": VERSION, "assembly_asserted": "GRCh38",
        "loop_file": str(Path(loop_path).resolve()),
        "gtf_file": str(Path(gtf_path).resolve()),
        "promoter_flank_bp": flank, "distal_nearest_tss_limit_bp": nearest_bp,
        "measured_genes": len(measured),
        "measured_genes_with_unique_tss": sum(len(entries) for _, entries in index.values()),
        "ambiguous_measured_symbols": ambiguous,
        "missing_measured_symbols": missing,
        "string_node_count": len(core), "perturbation_target_count": len(targets),
        "strict_promoter_promoter": graph_summary(strict, targets, core),
        "hypothesis_promoter_distal_nearest": graph_summary(hypothetical, targets, core),
        **stats,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("data")
    p.add_argument("string")
    p.add_argument("loops")
    p.add_argument("gtf")
    p.add_argument("--out", default="k562_hic_mapping_v2.json")
    p.add_argument("--promoter-flank", type=int, default=2000)
    p.add_argument("--distal-nearest-bp", type=int, default=50000)
    args = p.parse_args()
    if args.promoter_flank < 0 or args.distal_nearest_bp < 0:
        p.error("Mapping distances cannot be negative")
    for path in (args.data, args.string, args.loops, args.gtf):
        if not Path(path).is_file():
            p.error(f"Missing file: {path}")
    import scanpy as sc
    adata = sc.read_h5ad(args.data, backed="r")
    try:
        if adata.uns.get("normalization_protocol") != "2026-09-29-norman-cp10k-log1p-v2":
            raise ValueError("Expected verified normalized Norman data")
        measured = (adata.var["gene_symbol"].astype(str).tolist()
                    if "gene_symbol" in adata.var else adata.var_names.astype(str).tolist())
    finally:
        adata.file.close()
    with open(args.string, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("species") != 9606 or manifest.get("split") != "balanced1":
        raise ValueError("Expected human STRING manifest for balanced1")
    result = build_graph(measured, manifest, args.loops, args.gtf,
                         args.promoter_flank, args.distal_nearest_bp)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(f"Version={VERSION}; loops={result['loops_read']}; "
          f"measured genes with unique TSS="
          f"{result['measured_genes_with_unique_tss']}/{len(measured)}")
    print(f"Any measured promoter in {result['loops_with_any_measured_promoter']} "
          f"loops; both promoters in {result['loops_with_two_measured_promoters']}")
    for label in ("strict_promoter_promoter", "hypothesis_promoter_distal_nearest"):
        graph = result[label]
        print(f"{label}: edges={len(graph['edges'])}; "
              f"perturbation targets connected={graph['connected_perturbation_targets']}/"
              f"{result['perturbation_target_count']}; "
              f"STRING nodes connected={graph['connected_string_nodes']}/"
              f"{result['string_node_count']}")
    print(f"Saved {args.out}. Nearest-TSS edges are a proximity HYPOTHESIS, not "
          "validated enhancer-gene links. No perturbation outcomes were read.")


if __name__ == "__main__":
    main()

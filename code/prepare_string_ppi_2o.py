r"""Fetch a gene-aligned STRING prior for the fixed Norman unseen-target split.

PowerShell:
  python .\prepare_string_ppi_2o.py .\norman_processed.h5ad --split balanced1 --out .\string_balanced1.json

Uses STRING 12.0 human FUNCTIONAL associations by default. This is not a
physical-only interaction graph; use --network-type physical for that task.
STRING results are fetched by gene names, never by SVD coordinate indices.
No expression from held-out perturbed cells is used to construct the graph.
"""

import argparse
import hashlib
import json
import time

import numpy as np
import requests

import gene_graph_response_2o as graph
from count_balanced_graph_check_2o import predefined_splits, valid_targets


API = "https://version-12-0.string-db.org/api/tsv"


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def api_tsv(session, method, params):
    response = session.post(f"{API}/{method}", data=params, timeout=90)
    response.raise_for_status()
    lines = response.text.strip().splitlines()
    if not lines or "\t" not in lines[0]:
        raise ValueError(f"STRING {method} returned no TSV: {response.text[:250]}")
    import csv
    return list(csv.DictReader(lines, delimiter="\t"))


def fetch_graph(session, targets, measured, score, network_type, neighbors, pause):
    canonical = {g.upper(): g for g in measured}
    if len(canonical) != len(measured):
        raise ValueError("Expression gene symbols must be unique, ignoring case")
    mapped = {}
    aliases = {}
    for batch in batches(targets, 20):
        rows = api_tsv(session, "get_string_ids", {
            "identifiers": "\r".join(batch), "species": 9606,
            "echo_query": 1, "caller_identity": "iiitd_genetic_perturbation_research",
        })
        for row in rows:
            query = row.get("queryItem", "").upper()
            ident = row.get("stringId", "")
            if query not in {name.upper() for name in batch} or not ident:
                continue
            # STRING can match a synonym. Make the mapping visible for audit.
            gene = canonical.get(query)
            if gene is None or gene.upper() not in {name.upper() for name in targets}:
                continue
            previous = mapped.get(gene)
            if previous and previous != ident:
                raise ValueError(f"Ambiguous STRING mapping for {gene}: {previous} versus {ident}")
            mapped[gene] = ident
            if row.get("preferredName", "").upper() != gene.upper():
                aliases[gene] = row.get("preferredName", "")
        time.sleep(pause)
    reverse = {}
    for name, ident in mapped.items():
        if ident in reverse and reverse[ident] != name:
            raise ValueError(f"Two target genes map to the same STRING protein: {ident}")
        reverse[ident] = name
    edges = {}
    query_ids = list(reverse)
    for batch in batches(query_ids, 10):
        rows = api_tsv(session, "interaction_partners", {
            "identifiers": "\r".join(batch), "species": 9606,
            "limit": 20, "required_score": score, "network_type": network_type,
            "caller_identity": "iiitd_genetic_perturbation_research",
        })
        for row in rows:
            left = reverse.get(row.get("stringId_A", "")) or canonical.get(
                row.get("preferredName_A", "").upper())
            right = reverse.get(row.get("stringId_B", "")) or canonical.get(
                row.get("preferredName_B", "").upper())
            if not left or not right or left == right or (left not in mapped and right not in mapped):
                continue
            strength = float(row["score"])
            if not 0 <= strength <= 1:
                raise ValueError(f"Unexpected STRING combined score: {strength}")
            key = tuple(sorted((left, right)))
            edges[key] = max(edges.get(key, 0), strength)
        time.sleep(pause)
    core = set(targets)
    ranking = {}
    for (left, right), weight in edges.items():
        for candidate, partner in ((left, right), (right, left)):
            if candidate not in core and partner in core:
                count, total = ranking.get(candidate, (0, 0.0))
                ranking[candidate] = (count + 1, total + weight)
    selected = [name for name, _ in sorted(ranking.items(),
                key=lambda item: (-item[1][0], -item[1][1], item[0]))[:neighbors]]
    nodes = targets + selected
    permitted = set(nodes)
    kept = [[left, right, weight] for (left, right), weight in sorted(edges.items())
            if left in permitted and right in permitted]
    coverage = sorted(set(name for edge in kept for name in edge[:2]) & core)
    return nodes, kept, mapped, aliases, coverage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Norman processed .h5ad")
    parser.add_argument("--split", choices=("balanced1", "balanced2", "balanced3", "top20"),
                        default="balanced1")
    parser.add_argument("--out", default="string_balanced1.json")
    parser.add_argument("--network-type", choices=("functional", "physical"),
                        default="functional")
    parser.add_argument("--score", type=int, default=400)
    parser.add_argument("--neighbors", type=int, default=128,
                        help="Maximum extra measured-gene nodes beyond perturbation targets")
    parser.add_argument("--pause", type=float, default=0.3)
    args = parser.parse_args()
    if not 0 <= args.score <= 1000 or args.neighbors < 0 or args.pause < 0:
        parser.error("Invalid score, neighbors, or pause")
    try:
        import scanpy as sc
    except ImportError as exc:
        parser.error(f"Scanpy is required: {exc}")
    adata = sc.read_h5ad(args.data, backed="r")
    try:
        conditions = adata.obs["perturbation"].astype(str).to_numpy()
        genes = (adata.var["gene_symbol"].astype(str).tolist()
                 if "gene_symbol" in adata.var else adata.var_names.astype(str).tolist())
    finally:
        adata.file.close()
    if args.split == "top20":
        heldout, train_mask, test_mask = graph.split_conditions(conditions, genes, 20)
    else:
        split_no = int(args.split[-1])
        _, heldout, train_mask, test_mask = list(
            predefined_splits(conditions, genes, split_no))[-1]
    if np.any(train_mask & test_mask):
        raise ValueError("Train/test targets overlap")
    counts = valid_targets(conditions, genes)
    train_targets = sorted(name for name in counts if np.any(train_mask & (conditions == name)))
    targets = train_targets + list(heldout)
    print(f"Split={args.split}; training={len(train_targets)}; held-out={len(heldout)}; "
          f"STRING 12.0 {args.network_type} score>={args.score}", flush=True)
    with requests.Session() as session:
        nodes, edges, mapped, aliases, coverage = fetch_graph(
            session, targets, genes, args.score, args.network_type,
            args.neighbors, args.pause)
    result = {
        "source": API, "species": 9606, "network_type": args.network_type,
        "score_threshold": args.score, "split": args.split,
        "gene_sha256": hashlib.sha256("\n".join(genes).encode()).hexdigest(),
        "train_targets": train_targets, "heldout": list(heldout),
        "nodes": nodes, "edges": edges, "mapped": mapped,
        "different_preferred_names": aliases,
    }
    if not edges:
        raise ValueError("No mapped STRING edges; nothing saved. Check gene naming and API outputs")
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    covered_train = len(set(train_targets) & set(coverage))
    covered_test = len(set(heldout) & set(coverage))
    print(f"Saved {args.out}: mapped targets={len(mapped)}/{len(targets)}; "
          f"measured graph nodes={len(nodes)}, edges={len(edges)}", flush=True)
    print(f"Training targets with an edge={covered_train}/{len(train_targets)}; "
          f"held-out targets with an edge={covered_test}/{len(heldout)}", flush=True)
    print("Uncovered held-out targets: " + ", ".join(sorted(set(heldout)-set(coverage))),
          flush=True)
    if aliases:
        print("Review STRING synonym matches in different_preferred_names inside the JSON.",
              flush=True)
    print("No target perturbation outcomes were used in STRING construction.", flush=True)


if __name__ == "__main__":
    main()

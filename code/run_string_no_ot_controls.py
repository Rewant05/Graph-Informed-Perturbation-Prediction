"""Controlled no-OT follow-up; original implementation remains unchanged."""
import compare_pair_ot_vae_ddpm_gears as base

_original_graph_inputs = base.graph_inputs

def graph_inputs(genes, pairs, string, hic, max_nodes, seed):
    nodes, modes, support = _original_graph_inputs(genes, pairs, string, hic, max_nodes, seed)
    # Reuse the exact permutation used by the dual shuffle, keeping Hi-C absent.
    shuffled_string, _ = modes["shuffled"]
    zero, _ = modes["no_edges"]
    modes["shuffled_string"] = (shuffled_string, zero)
    return nodes, modes, support

base.graph_inputs = graph_inputs
base.VERSION = "2026-10-07-string-no-ot-controls-v1"
base.VARIANTS = ("no_edges:no_ot", "shuffled_string:no_ot", "string:no_ot")

if __name__ == "__main__":
    base.main()

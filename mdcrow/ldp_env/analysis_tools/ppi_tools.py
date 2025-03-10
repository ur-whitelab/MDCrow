import itertools
from typing import Optional

import mdtraj as md
import numpy as np
from state import MDCrowState


def ppi_distance(file_path, binding_site="protein"):
    """
    Calculates minimum heavy-atom distance between peptide (assumed to be
    the smallest chain) and protein, considering a specified binding site.
    Returns the average distance between these two.

    Can work with any protein-protein interaction (PPI).
    """
    traj = md.load(file_path)
    if traj.topology.n_chains == 1:
        raise ValueError("Only one chain found. Cannot compute PPI distance.")

    # get the smallest chain
    peptide_idx = np.argmin([chain.n_residues for chain in traj.topology.chains])
    peptide_residues = {r.index for r in traj.topology.chain(peptide_idx).residues}

    # get protein residues
    protein_atoms = traj.topology.select(
        f"({binding_site}) and not chainid {peptide_idx}"
    )
    protein_residues = {traj.topology.atom(a).residue.index for a in protein_atoms}
    if len(protein_residues) == 0:
        raise ValueError("No matching residues found for the binding site.")

    res_pairs = list(itertools.product(peptide_residues, protein_residues))
    res_pairs_array = np.array(res_pairs)
    all_d, _ = md.compute_contacts(traj, res_pairs_array, scheme="closest-heavy")
    if all_d.size > 0:
        avg_dist = np.mean(all_d)
        return avg_dist
    else:
        raise ValueError("For unknown reason, no distances between contacts found.")


def compute_ppi_distance(
    state: MDCrowState, pdb_file: str, binding_site: Optional[str] = None
):
    """
    Computes the distance between protein-protein interaction (PPI) sites.

    Args:
        pdb_file (str): File ID of the PDB file containing the protein-protein interaction.
        binding_site (Optional[str], optional): List of selected residues defining
            the binding site of the protein, using MDTraj selection syntax.
    """

    if not state.path_registry:
        return "Failed. Error: Path registry is not set", 0, False
    file_path = state.path_registry.get_mapped_path(pdb_file)
    if not file_path:
        return f"Failed. File not found: {pdb_file}", 0, False
    if not file_path.endswith(".pdb"):
        return "Failed. Error with input: PDB file must have .pdb extension", 0, False
    try:
        avg_dist = ppi_distance(file_path, binding_site=binding_site)
    except Exception as e:
        return f"Failed. Something went wrong. {type(e).__name__}: {e}", 0, False
    return f"Succeeded. PPI average distance is {avg_dist}\n", 0, False

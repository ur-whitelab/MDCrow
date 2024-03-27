import os
from typing import Optional

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry

#I think the forcefield tool is missing. I need amber, gromacs,
# itp to be added

class SaltBridgeFunction:
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.includes_top = [".h5", ".lh5", ".pdb"]

    def find_salt_bridges(traj_file, top_file, threshold_distance=0.4, residue_pairs=None):
        salt_bridges = []
        # Load trajectory using MDTraj
        traj = md.load(traj_file, top= top_file)
        if residue_pairs is None:
            residue_pairs = [("ARG", "ASP"), ("ARG", "GLU"), ("LYS", "ASP"), ("LYS", "GLU")]

        for pair in residue_pairs:
            donor_residues = traj.topology.select(f'residue_name == "{pair[0]}"')
            acceptor_residues = traj.topology.select(f'residue_name == "{pair[1]}"')

            for donor_idx in donor_residues:
                for acceptor_idx in acceptor_residues:
                    distances = md.compute_distances(traj, [[donor_idx, acceptor_idx]])
                    if any(d <= threshold_distance for d in distances):
                        salt_bridges.append((donor_idx, acceptor_idx))

        return salt_bridges


# Load trajectory using MDTraj
traj = md.load("trajectory.dcd", top="topology.pdb")

# Perform salt bridge analysis
salt_bridges = find_salt_bridges(traj)

# Print identified salt bridges
print("Salt bridges found:")
for bridge in salt_bridges:
    print(
        f"Residue {traj.topology.atom(bridge[0]).residue.index + 1} ({traj.topology.atom(bridge[0]).residue.name}) - "
        f"Residue {traj.topology.atom(bridge[1]).residue.index + 1} ({traj.topology.atom(bridge[1]).residue.name})"
    )

    
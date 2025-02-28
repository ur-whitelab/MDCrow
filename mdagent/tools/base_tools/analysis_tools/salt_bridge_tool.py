import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import pandas as pd
from langchain.tools import BaseTool

from mdagent.utils import FileType, PathRegistry, load_single_traj, save_plot


class SaltBridgeFunction:
    def __init__(self, path_registry):
        self.path_registry = path_registry
        self.salt_bridge_data = []  # stores paired salt bridges
        self.salt_bridge_counts = []
        self.traj = None
        self.traj_file = ""
        self.specific_frame = None
        self.pH = 7.0

    def _load_traj(self, traj_file, top_file):
        self.traj = load_single_traj(
            self.path_registry, top_fileid=top_file, traj_fileid=traj_file
        )
        self.traj_file = traj_file if traj_file else top_file

    def find_salt_bridges(
        self,
        threshold_distance: float = 0.4,
        pH_dependence=False,
        target_pH=7.0,
        specific_frame=None,
        acidic_residues=("ASP", "GLU"),
        basic_residues=("ARG", "LYS", "HIS"),
        include_terminals=True,
    ):
        """
        Analyze salt bridges in a molecular structure or trajectory.

        Parameters:
        - threshold_distance (float): Distance cutoff (Å) to define a salt bridge.
        - pH_dependence (bool): Whether to consider pH-based protonation states.
        - target_pH (float): The pH level to consider when adjusting protonation states.
        - specific_frame (int, optional): To compute salt bridge details for that frame.
        - acidic_residues (tuple): List of acidic residues (default: ASP, GLU).
        - basic_residues (tuple): List of basic residues (default: ARG, LYS, HIS).
        - include_terminals (bool): Whether to include N-/C-termini in salt bridges.

        Returns:
        - If trajectory: Plots salt bridge counts over time.
        - If single frame or specific frame: Saves detailed salt bridge list.
        """
        if self.traj is None:
            raise Exception("MDTrajectory hasn't been loaded")
        self.pH = target_pH
        self.specific_frame = specific_frame

        acidic_selection = (
            f"(resname {' '.join(acidic_residues)}) and name OE1 OE2 OD1 OD2"
        )
        basic_selection = f"(resname {' '.join(basic_residues)}) and name NZ NH1 NH2 NE"

        if include_terminals:
            acidic_selection += " or name OXT"  # C-terminal oxygen
            basic_selection += " or (resid 1 and name N)"  # N-terminal amine

        acidic_atoms = self.traj.topology.select(acidic_selection)
        basic_atoms = self.traj.topology.select(basic_selection)

        if pH_dependence:
            # pKa values from https://dx.doi.org/10.1146/annurev-biophys-083012-130351
            # future improvements: get pKa with PROPKA?
            pKa_values = {
                "ASP": 4.0,
                "GLU": 4.4,
                "HIS": 6.8,
                "LYS": 10.4,
                "ARG": 13.5,
                "CYS": 8.3,
                "TYR": 9.6,
                "N-Terminus": 8.0,
                "C-Terminus": 3.6,
            }
            acidic_atoms = [
                a
                for a in acidic_atoms
                if (pKa := pKa_values.get(self.traj.topology.atom(a).residue.name))
                is not None
                and target_pH > pKa
            ]

            basic_atoms = [
                a
                for a in basic_atoms
                if (pKa := pKa_values.get(self.traj.topology.atom(a).residue.name))
                is not None
                and target_pH < pKa
            ]

        acidic_atoms = np.array(acidic_atoms)
        basic_atoms = np.array(basic_atoms)

        if acidic_atoms.size == 0 or basic_atoms.size == 0:
            return None

        # Compute distances
        pairs = np.array(np.meshgrid(acidic_atoms, basic_atoms)).T.reshape(-1, 2)
        all_distances = md.compute_distances(self.traj, pairs)

        salt_bridge_counts = []
        salt_bridge_data = []
        for frame_idx in range(self.traj.n_frames):
            within_cutoff = all_distances[frame_idx] < threshold_distance
            count = np.sum(within_cutoff)
            salt_bridge_counts.append(count)

            valid_indices = np.where(within_cutoff)[0]
            filtered_pairs = pairs[valid_indices]
            for pair_idx, (acid_idx, base_idx) in zip(valid_indices, filtered_pairs):
                acid_atom = self.traj.topology.atom(acid_idx)
                base_atom = self.traj.topology.atom(base_idx)
                distance_value = all_distances[frame_idx, pair_idx]
                salt_bridge_data.append(
                    {
                        "Frame": frame_idx,
                        "Residue1": f"{acid_atom.residue.name}{acid_atom.residue.index}",
                        "Atom1": acid_atom.name,
                        "Residue2": f"{base_atom.residue.name}{base_atom.residue.index}",
                        "Atom2": base_atom.name,
                        "Distance (Å)": f"{distance_value:.4f}",
                    }
                )
        self.salt_bridge_counts = salt_bridge_counts
        self.salt_bridge_data = salt_bridge_data

    def plot_salt_bridge_counts(self):
        if not self.salt_bridge_data or self.traj.n_frames == 1 or self.specific_frame:
            print("i'm here")
            print(not self.salt_bridge_data)
            print(not self.traj.n_frames)
            print(not self.specific_frame)
            return None

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(self.traj.n_frames),
            self.salt_bridge_counts,
            marker="o",
            linestyle="-",
            color="b",
        )
        plt.title(f"Salt Bridge Count Over Time - {self.traj_file}")
        plt.xlabel("Frame")
        plt.ylabel("Total Salt Bridge Count")
        plt.grid(True)
        fig_id = save_plot(
            self.path_registry,
            "salt_bridge",
            f"figure of salt bridge counts for {self.traj_file} with pH {self.pH}",
        )
        plt.close()
        return fig_id

    def save_results_to_file(self):
        if self.traj is None:
            raise Exception("Trajectory is None")
        if not self.salt_bridge_data:
            return None

        frame_to_save = 0 if self.traj.n_frames == 1 else self.specific_frame
        if frame_to_save is not None and 0 <= frame_to_save < self.traj.n_frames:
            num_sb = self.salt_bridge_counts[frame_to_save]
            print(
                f"We found {num_sb} salt bridges for {self.traj_file} in frame {frame_to_save}."
            )
            filtered_salt_bridge_data = [
                entry
                for entry in self.salt_bridge_data
                if entry["Frame"] == frame_to_save
            ]
            if not filtered_salt_bridge_data:
                return
            df = pd.DataFrame(filtered_salt_bridge_data)
        else:
            df = pd.DataFrame(self.salt_bridge_data)

        # save to file, add to path registry
        file_name = self.path_registry.write_file_name(
            FileType.RECORD,
            record_type="salt_bridges",
            file_format="csv",
        )
        file_id = self.path_registry.get_fileid(file_name, FileType.RECORD)
        file_path = f"{self.path_registry.ckpt_records}/{file_name}"
        df.to_csv(file_path, index=False)
        desc = f"salt bridge analysis for {self.traj_file} in frame {frame_to_save}"
        self.path_registry.map_path(file_id, file_path, description=desc)
        return file_id

    def compute_salt_bridges(
        self,
        traj_file,
        top_file,
        threshold_distance,
        pH_dependence,
        target_pH,
        specific_frame,
        acidic_residues,
        basic_residues,
        include_terminals,
    ):
        self._load_traj(traj_file, top_file)
        self.find_salt_bridges(
            threshold_distance,
            pH_dependence,
            target_pH,
            specific_frame,
            acidic_residues,
            basic_residues,
            include_terminals,
        )
        file_id = self.save_results_to_file()
        fig_id = self.plot_salt_bridge_counts()
        return file_id, fig_id


class SaltBridgeTool(BaseTool):
    name = "SaltBridgeTool"
    description = (
        "A tool to find and count salt bridges in a protein trajectory. "
        "You need to provide either PDB file or trajectory and topology files. "
        "The rest of inputs are optional: threshold_distance (default: 0.4), "
        "pH_dependence (whether to consider pH-based protonation states), "
        "target_pH (default: 7.0), specific_frame, acidic_residues "
        "(default: ('ASP', 'GLU')), basic_residues (default: ('ARG', 'LYS', 'HIS')), "
        "and include_terminals (default: True)."
    )
    path_registry: PathRegistry | None = None

    def __init__(self, path_registry):
        super().__init__()
        self.path_registry = path_registry

    def _run(
        self,
        traj_file: str,
        top_file: str | None = None,
        threshold_distance=0.4,
        pH_dependence=False,
        target_pH=7.0,
        specific_frame=None,
        acidic_residues=("ASP", "GLU"),
        basic_residues=("ARG", "LYS", "HIS"),
        include_terminals=True,
    ):
        try:
            if self.path_registry is None:
                return "Path registry is not set"

            salt_bridge_function = SaltBridgeFunction(self.path_registry)
            results_file_id, fig_id = salt_bridge_function.compute_salt_bridges(
                traj_file,
                top_file,
                threshold_distance,
                pH_dependence,
                target_pH,
                specific_frame,
                acidic_residues,
                basic_residues,
                include_terminals,
            )
            if not results_file_id:
                return (
                    "Succeeded. No salt bridges are found in "
                    f"{salt_bridge_function.traj_file}."
                )

            message = f"Saved results with file id: {results_file_id} "
            if fig_id:
                message += f"and figure with fig id {fig_id}."
            return "Succeeded. " + message
        except Exception as e:
            return f"Failed. {type(e).__name__}: {e}"

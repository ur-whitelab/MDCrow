import os
import shutil


class SetCheckpoint:
    def find_root_dir(self):
        current_dir = os.getcwd()
        while current_dir != "/":
            if "setup.py" in os.listdir(current_dir):
                return os.path.abspath(current_dir)
            else:
                current_dir = os.path.dirname(current_dir)
        return None

    def make_ckpt_parent_folder(self, ckpt_dir: str = "ckpt"):
        root = self.find_root_dir()
        if not root:
            raise ValueError("Root directory not found.")
        ckpt_path = os.path.join(root, ckpt_dir)
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        return ckpt_path

    def set_ckpt_subdir(self, ckpt_dir: str = "ckpt", ckpt_parent_folder: str = "ckpt"):
        if ckpt_dir != "ckpt":
            ckpt_parent_path = self.make_ckpt_parent_folder(ckpt_parent_folder)
            ckpt_subdir = os.path.join(ckpt_parent_path, f"{ckpt_dir}")
            os.makedirs(ckpt_subdir, exist_ok=True)
            return ckpt_subdir
        ckpt_parent_path = self.make_ckpt_parent_folder(ckpt_parent_folder)
        ckpt_iter = 0
        ckpt_subdir = os.path.join(ckpt_parent_path, f"{ckpt_dir}_{ckpt_iter}")
        while os.path.exists(ckpt_subdir):
            ckpt_subdir = os.path.join(ckpt_parent_path, f"{ckpt_dir}_{ckpt_iter}")
            ckpt_iter += 1
        os.makedirs(ckpt_subdir, exist_ok=True)
        return ckpt_subdir

    def clear_ckpt(
        self,
        ckpt_dir: str = "ckpt",
        parent_ckpt_path: str = "ckpt",
        keep_root: bool = False,
    ):
        parent_ckpt_path = self.make_ckpt_parent_folder(parent_ckpt_path)
        ckpt_path = os.path.join(parent_ckpt_path, ckpt_dir)
        if not os.path.exists(ckpt_path):
            return None
        shutil.rmtree(ckpt_path)
        if keep_root:
            os.makedirs(ckpt_path)
        return None

    def clear_all_ckpts(self, parent_ckpt_path: str = "ckpt", keep_root: bool = False):
        parent_ckpt_path = self.make_ckpt_parent_folder(parent_ckpt_path)
        shutil.rmtree(parent_ckpt_path)
        if keep_root:
            _ = self.make_ckpt_parent_folder(parent_ckpt_path)
        return None

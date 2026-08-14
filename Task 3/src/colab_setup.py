"""Environment setup for local Jupyter and Google Colab."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def setup_environment(
    task_folder: str = "Task 3",
    extra_packages: list[str] | None = None,
) -> Path:
    """
    Configure paths and dependencies for the current runtime.

    On Google Colab:
      - Installs required pip packages
      - Mounts Google Drive
      - Locates the task folder under common Drive/content paths

    Locally:
      - Resolves project root from cwd (handles notebooks/ subfolder)
    """
    in_colab = "google.colab" in sys.modules

    packages = ["pandas", "numpy", "scikit-learn", "matplotlib", "seaborn"]
    if extra_packages:
        packages.extend(extra_packages)

    if in_colab:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", *packages],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        from google.colab import drive

        drive.mount("/content/drive", force_remount=False)

        candidates = [
            Path(f"/content/drive/MyDrive/Thiranex/{task_folder}"),
            Path(f"/content/drive/MyDrive/{task_folder}"),
            Path(f"/content/Thiranex/{task_folder}"),
            Path(f"/content/{task_folder}"),
        ]
        project_root = next((p for p in candidates if p.exists()), None)
        if project_root is None:
            raise FileNotFoundError(
                f"Could not find '{task_folder}'.\n"
                f"Upload the folder to Google Drive at MyDrive/Thiranex/{task_folder}\n"
                f"or clone your repo to /content/Thiranex/{task_folder}."
            )
    else:
        project_root = Path.cwd().resolve()
        if project_root.name == "notebooks":
            project_root = project_root.parent

    os.chdir(project_root)

    src_path = project_root / "src"
    if src_path.exists() and str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    return project_root


def resolve_data_path(project_root: Path, relative_path: str, fallback_paths: list[str] | None = None) -> Path:
    """Resolve a data file within the project or sibling task folders."""
    primary = project_root / relative_path
    if primary.exists():
        return primary

    if fallback_paths:
        for fallback in fallback_paths:
            candidate = project_root.parent / fallback
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Data file not found: {relative_path}\n"
        f"Checked: {primary}"
        + (f" and fallbacks: {fallback_paths}" if fallback_paths else "")
    )

from pathlib import Path
import os
import subprocess

def _git_root():
    try:
        root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True, capture_output=True, text=True
        ).stdout.strip()
        return Path(root)
    except Exception:
        return None

def project_root(marker_files=(".git", "pyproject.toml", "requirements.txt")):
    # Try git first
    root = _git_root()
    if root and root.exists():
        return root
    # Fallback: walk up from CWD until a marker is found
    p = Path.cwd()
    for parent in [p, *p.parents]:
        if any((parent / m).exists() for m in marker_files):
            return parent
    return p  # last resort: current dir

def data_dir():
    # 1) env var override (great for Colab/Drive or different machines)
    env_path = os.getenv("DATA_DIR")
    if env_path:
        return Path(env_path)
    # 2) default: repo_root/data
    return project_root() / "data"

"""Where this project lives: installers download Omnicompute from here, and the permanent mesh
link is a page on its GitHub Pages site. Forks change these two lines (or set the variables)."""

import os
from pathlib import Path

REPO = os.environ.get("OMNICOMPUTE_REPO", "TOS-main/omnicompute")
PAGES = os.environ.get("OMNICOMPUTE_PAGES", f"https://{REPO.split('/')[0].lower()}.github.io/{REPO.split('/')[1]}")
# The project creator's own mesh, which anyone may help with spare compute: installers offer it
# when no code is given, and `omnicompute contribute creator` joins it. Forks put their own here.
CREATOR_CODE = os.environ.get("OMNICOMPUTE_CREATOR", "cfb109c420aa15d6f1ad769a9")
ARCHIVE = f"https://github.com/{REPO}/archive/refs/heads/main.zip"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"


def permanent_link(key: str, full: bool = False) -> str:
    """A link that always finds the mesh's host, wherever it moved (see beacon.py). The key sits
    after '#', so it never leaves the browser."""
    return f"{PAGES}/go.html#{'k' if full else 'c'}={key}"


def home_dir() -> Path:
    """Where keys, limits, chats and models live. Computers that ran EdgeMesh (this project's
    old name) keep using ~/.edgemesh, so their mesh key and links stay the same."""
    if os.environ.get("OMNICOMPUTE_HOME"):
        return Path(os.environ["OMNICOMPUTE_HOME"])
    legacy = Path.home() / ".edgemesh"
    return legacy if (legacy / "mesh.key").exists() else Path.home() / ".omnicompute"

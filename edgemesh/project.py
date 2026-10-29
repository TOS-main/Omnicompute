"""Where this project lives: installers download EdgeMesh from here, and the permanent mesh
link is a page on its GitHub Pages site. Forks change these two lines (or set the variables)."""

import os

REPO = os.environ.get("EDGEMESH_REPO", "TOS-main/edgemesh")
PAGES = os.environ.get("EDGEMESH_PAGES", f"https://{REPO.split('/')[0].lower()}.github.io/{REPO.split('/')[1]}")
# The project creator's own mesh, which anyone may help with spare compute: installers offer it
# when no code is given, and `edgemesh contribute creator` joins it. Forks put their own here.
CREATOR_CODE = os.environ.get("EDGEMESH_CREATOR", "cfb109c420aa15d6f1ad769a9")
ARCHIVE = f"https://github.com/{REPO}/archive/refs/heads/main.zip"
RAW = f"https://raw.githubusercontent.com/{REPO}/main"


def permanent_link(key: str, full: bool = False) -> str:
    """A link that always finds the mesh's host, wherever it moved (see beacon.py). The key sits
    after '#', so it never leaves the browser."""
    return f"{PAGES}/go.html#{'k' if full else 'c'}={key}"

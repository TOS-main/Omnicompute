"""Build a one-file Omnicompute app for the current OS (Windows .exe, macOS, Linux) with PyInstaller.

    pip install pyinstaller .
    python scripts/build_app.py      # -> dist/omnicompute[.exe]

Double-clicking the result asks "host or join", so there is nothing to install.
"""

import os
import sys

import PyInstaller.__main__

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sep = ";" if sys.platform == "win32" else ":"
PyInstaller.__main__.run([
    os.path.join(ROOT, "scripts", "app_entry.py"),
    "--onefile", "--name", "omnicompute", "--console",
    "--add-data", f"{os.path.join(ROOT, 'omnicompute', 'web')}{sep}omnicompute/web",
    "--paths", ROOT, "--collect-submodules", "omnicompute",
    "--distpath", os.path.join(ROOT, "dist"), "--workpath", os.path.join(ROOT, "build"),
    "--specpath", os.path.join(ROOT, "build"),
])

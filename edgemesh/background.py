"""`edgemesh contribute LINK`: keep this computer contributing quietly in the background.

Starts `edgemesh join` as a hidden background process (no console window on Windows), writes
its output to ~/.edgemesh/contribute.log, and registers it to start again at login:

  * Windows: a "Run" entry in the current user's registry (pythonw, so no window appears)
  * macOS:   a LaunchAgent in ~/Library/LaunchAgents
  * Linux:   an XDG autostart entry in ~/.config/autostart

`edgemesh contribute --stop` stops it and removes the autostart entry. The main computer can
also remove a device from the mesh (Devices tab): the device then deletes its autostart entry
and, if the one-line installer put it there, its whole EdgeMesh folder. Usage limits still
apply (default 80%, or set them from the main computer's Devices tab).
"""

from __future__ import annotations

import os
import platform
import signal
import subprocess
import sys
from pathlib import Path

LABEL = "com.edgemesh.contribute"
MARKER = "installed-by-edgemesh-installer"  # written by install.ps1 / install.sh


def _python(windowless: bool) -> str:
    exe = Path(sys.executable)
    if windowless and platform.system() == "Windows":
        w = exe.with_name("pythonw.exe")
        if w.exists():
            return str(w)
    return str(exe)


def join_command(link: str, name: str, home: Path, extra: list[str], windowless: bool = True) -> list[str]:
    return [_python(windowless), "-m", "edgemesh", "join", link, "--name", name,
            "--log", str(home / "contribute.log"), *extra]


def start(link: str, name: str, home: Path, extra: list[str], autostart: bool = True) -> int:
    """Start the hidden background contributor; returns its process id."""
    stop(home, quiet=True, keep_autostart=True)
    home.mkdir(parents=True, exist_ok=True)
    cmd = join_command(link, name, home, extra)
    kw: dict = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
                "cwd": str(home)}
    if platform.system() == "Windows":
        kw["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED, NEW_PROCESS_GROUP, NO_WINDOW
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(cmd, **kw)
    (home / "contribute.pid").write_text(str(proc.pid))
    if autostart:
        install_autostart(cmd)
    return proc.pid


def stop(home: Path, quiet: bool = False, keep_autostart: bool = False) -> bool:
    pid_file = home / "contribute.pid"
    stopped = False
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if platform.system() == "Windows":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
            else:
                os.kill(pid, signal.SIGTERM)
            stopped = True
        except (ValueError, OSError):
            pass
        pid_file.unlink(missing_ok=True)
    if not keep_autostart:
        remove_autostart()
    if not quiet:
        print("Stopped contributing." if stopped else "EdgeMesh was not running in the background.")
    return stopped


def _quote_windows(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def install_autostart(cmd: list[str]) -> str:
    system = platform.system()
    if system == "Windows":
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                            winreg.KEY_SET_VALUE) as k:
            winreg.SetValueEx(k, "EdgeMesh", 0, winreg.REG_SZ, _quote_windows(cmd))
        return "registry Run key"
    if system == "Darwin":
        from xml.sax.saxutils import escape

        args = "".join(f"<string>{escape(a)}</string>" for a in cmd)
        plist = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"
        plist.parent.mkdir(parents=True, exist_ok=True)
        plist.write_text(f'<?xml version="1.0" encoding="UTF-8"?>\n<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
                         f'"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n<plist version="1.0"><dict>'
                         f"<key>Label</key><string>{LABEL}</string><key>ProgramArguments</key><array>{args}</array>"
                         f"<key>RunAtLoad</key><true/><key>ProcessType</key><string>Background</string></dict></plist>\n")
        return str(plist)
    import shlex

    script = Path(cmd[cmd.index("--log") + 1]).with_name("contribute.sh")
    script.write_text("#!/bin/sh\nexec " + shlex.join(cmd) + " >/dev/null 2>&1\n")
    desktop = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "autostart" / "edgemesh.desktop"
    desktop.parent.mkdir(parents=True, exist_ok=True)
    desktop.write_text(f"[Desktop Entry]\nType=Application\nName=EdgeMesh contributor\nExec=sh {script}\n"
                       "NoDisplay=true\nX-GNOME-Autostart-enabled=true\n")
    return str(desktop)


def remove_autostart() -> None:
    system = platform.system()
    try:
        if system == "Windows":
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0,
                                winreg.KEY_SET_VALUE) as k:
                winreg.DeleteValue(k, "EdgeMesh")
        elif system == "Darwin":
            (Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist").unlink(missing_ok=True)
        else:
            (Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "autostart" / "edgemesh.desktop").unlink(missing_ok=True)
    except OSError:
        pass


def uninstall(home: Path) -> list[str]:
    """Remove autostart and, for installer-made setups, delete EdgeMesh's folders once this
    process has exited. Returns the folders scheduled for deletion."""
    remove_autostart()
    (home / "contribute.pid").unlink(missing_ok=True)
    folders = []
    install_dir = Path(sys.prefix).parent  # the installer puts the venv inside its folder
    if (install_dir / MARKER).exists():
        folders.append(install_dir)
        if home.exists() and home.resolve() != install_dir.resolve():
            folders.append(home)
    if not folders:
        return []
    pid = os.getpid()
    if platform.system() == "Windows":
        dirs = " ".join(f'"{f}"' for f in folders)
        cmd = f'cmd /c "ping -n 4 127.0.0.1 >nul & rmdir /s /q {dirs}"'
        # wait for this process to exit (it does within a second), then delete
        subprocess.Popen(cmd, creationflags=0x00000008 | 0x08000000, close_fds=True, cwd=os.environ.get("TEMP", "C:\\"))
    else:
        import shlex

        dirs = " ".join(shlex.quote(str(f)) for f in folders)
        subprocess.Popen(["sh", "-c", f"while kill -0 {pid} 2>/dev/null; do sleep 0.5; done; rm -rf {dirs}"],
                         start_new_session=True, cwd="/", stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    return [str(f) for f in folders]

# Build with: py -m PyInstaller --clean --noconfirm AutoAnki.spec

from pathlib import Path


project_root = Path(SPECPATH)

a = Analysis(
    [str(project_root / "src" / "gui.py")],
    pathex=[str(project_root / "src")],
    binaries=[],
    datas=[
        (str(project_root / "input" / "prompts"), "input/prompts"),
        (
            str(project_root / "input" / "prompt_components"),
            "input/prompt_components",
        ),
        (str(project_root / "input" / "words"), "input"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["unittest"],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="AutoAnki",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

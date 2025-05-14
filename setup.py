"""
setup.py for the *tampura* package.

• Runs SymK’s own build.py (CMake + Make) during `build_ext`.
• Copies everything produced in tampura/third_party/symk/builds/release/**
  into the wheel so `import tampura` can find the binaries at runtime.
• Works for both normal installs (`pip install .`, `pip install git+…`)
  and editable installs (`pip install -e .`).

Your pyproject.toml already contains the metadata (dependencies, etc.),
so this file only handles the custom build step.
"""

from pathlib import Path
import shutil
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

PKG_NAME = "tampura"
SYMK_SRC = Path(__file__).parent / PKG_NAME / "third_party" / "symk"
SYMK_OUT = SYMK_SRC / "builds" / "release"           # <- build.py final output


def run_symk_build() -> None:
    """Invoke SymK’s own build script (CMake ➜ Make)."""
    subprocess.check_call([sys.executable, "build.py"], cwd=SYMK_SRC)


class BuildExt(_build_ext):
    """Hook that compiles SymK and copies the artefacts into the wheel."""

    def run(self) -> None:
        # 1. Run the normal build_ext logic (even though we only have a dummy ext).
        super().run()

        # 2. Compile SymK.
        run_symk_build()

        # 3. Copy every file produced in builds/release/ into the build_lib tree
        #    so it ends up inside the wheel.
        dest_root = (
            Path(self.build_lib)
            / PKG_NAME
            / "third_party"
            / "symk"
            / "builds"
            / "release"
        )
        shutil.copytree(SYMK_OUT, dest_root, dirs_exist_ok=True)


# Dummy Extension – forces setuptools to execute build_ext even though the
# Python package itself has no C/C++ extensions.
dummy_ext = Extension(f"{PKG_NAME}._dummy", sources=[])

setup(
    name=PKG_NAME,
    version="0.1.0",
    packages=[PKG_NAME],
    ext_modules=[dummy_ext],
    cmdclass={"build_ext": BuildExt},
    # Ship everything we just copied:
    include_package_data=True,
    package_data={f"{PKG_NAME}.third_party.symk": ["builds/release/**/*"]},
    zip_safe=False,  # the wheel contains platform-specific binaries
)

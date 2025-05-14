# setup.py
from pathlib import Path
import subprocess, sys, shutil
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext

PKG_ROOT   = Path(__file__).parent
SYMK_DIR   = PKG_ROOT / "tampura" / "third_party" / "symk"
OUTPUT_DIR = SYMK_DIR / "builds"                 # whatever build.py writes to

def run_symk():
    subprocess.check_call([sys.executable, "build.py"], cwd=SYMK_DIR)

class BuildExt(_build_ext):
    def run(self):
        # 1. run Cython/C, Rust, etc. if you declared real Extension() objects
        super().run()

        # 2. run your own compiler
        run_symk()

        # 3. copy artefacts into the directory that will be packed into the wheel
        dest = Path(self.build_lib) / "tampura" / "third_party" / "symk" / "builds"
        dest.mkdir(parents=True, exist_ok=True)
        for path in OUTPUT_DIR.glob("*"):
            shutil.copy2(path, dest)

# --Dummy Extension so setuptools activates build_ext even if you have no *.c files
dummy = Extension("tampura._dummy", sources=[])

setup(
    name="tampura",
    version="0.1.0",
    packages=["tampura"],
    ext_modules=[dummy],          # ensures build_ext is part of the build chain
    cmdclass={"build_ext": BuildExt},
    package_data={"tampura.third_party.symk": ["builds/*"]},
)

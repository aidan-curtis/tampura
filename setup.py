from setuptools import setup
from setuptools.command.build_ext import build_ext as _build_ext
import subprocess
import os


class CustomBuildExt(_build_ext):
    def run(self):
        symk_dir = os.path.join(os.path.dirname(__file__), "tampura", "third_party", "symk")
        subprocess.check_call(["python", "build.py"], cwd=symk_dir)
        super().run()


setup(
    cmdclass={
        "build_ext": CustomBuildExt,
    }
)

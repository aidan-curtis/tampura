from setuptools import setup
from setuptools.command.build_py import build_py as _build_py
import subprocess
import os


class CustomBuildPy(_build_py):
    def run(self):
        # Run the original build_py first
        super().run()

        # Now run your custom script
        symk_dir = os.path.join(os.path.dirname(__file__), "third_party", "symk")
        subprocess.check_call(["python", "build.py"], cwd=symk_dir)


setup(
    name="tampura",
    version="0.1.0",
    packages=["tampura"],
    cmdclass={
        'build_py': CustomBuildPy,
    },
)

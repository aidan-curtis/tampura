"""Setup script."""

from setuptools import setup

setup(
    name="tampura",
    version="0.1.0",
    packages=["tampura"],
    include_package_data=True,
    install_requires=[
        "black==23.9.1",
        "docformatter==1.7.5",
        "isort==5.12.0",
        "pylint>=2.14.5",
        "pytest-pylint>=0.18.0",
        "six",
        "pymdptoolbox",
        "pybullet",
        "rtree",
        "trimesh==3.21.7",
        "pyyaml",
        "pexpect",
        "tabulate",
        "zmq",
        "open3d",
        "gymnasium",
        "stable_baselines3",
        "tensorboard",
    ],
)

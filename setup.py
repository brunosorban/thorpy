from setuptools import setup, find_packages

setup(
    name="thorpy",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "sympy",
        "mayavi",
        "imageio",
        "casadi",
    ],
    # Metadata
    author="Bruno Sorban",
    author_email="brunosorban@gmail.com",
    description="This project comprises a trajectory generation algorithm with hardware limitaion constraints and a MPC controller to track the trajectory.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/brunosorban/hopper_traj",
)

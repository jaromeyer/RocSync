from setuptools import find_packages, setup

setup(
    name="rocsync",
    version="0.1.0",
    author="Jaro Meyer",
    author_email="jaro.meyer@protonmail.com",
    description="RocSync: Temporal Multi-Camera Synchronization",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jaromeyer/rocsync",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "matplotlib",
        "numpy",
        "opencv-python",
        "scikit-learn",
        "tqdm",
    ],
    entry_points={
        "console_scripts": [
            "rocsync=rocsync.main:main",
        ],
    },
)

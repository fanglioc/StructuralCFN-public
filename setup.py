from setuptools import setup, find_packages

setup(
    name="scfn",
    version="1.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xgboost",
        "tqdm",
        "pytorch-tabnet",
        "optuna",
    ],
    author="Fang Li",
    description="Structural Compositional Function Networks for Tabular Data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/StructuralCFN",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="minionbench",
    version="0.1.0",
    author="Harsh Singh",
    description="A benchmarking framework for evaluating language model performance across different configurations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/singhh5050/minionbench",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
    ],
    entry_points={
        "console_scripts": [
            "minionbench-sweep=minionbench.sweep_runner:main_sweep",
        ],
    },
) 
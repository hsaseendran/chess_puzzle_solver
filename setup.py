from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="chess-puzzle-solver-proven",
    version="1.0.0",
    author="Chess AI Team",
    author_email="chess-ai@example.com",
    description="A chess puzzle solver using proven neural network architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/chess-puzzle-solver-proven",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Games/Entertainment :: Board Games",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "pandas>=1.5.0",
        "python-chess>=1.9.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
        "matplotlib>=3.7.0",
        "scikit-learn>=1.2.0",
    ],
    entry_points={
        "console_scripts": [
            "train-puzzle-solver=scripts.train:main",
            "evaluate-puzzle-solver=scripts.evaluate:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
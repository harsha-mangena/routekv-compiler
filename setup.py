from setuptools import setup, find_packages

setup(
    name="routekv-compiler",
    version="0.1.0",
    description="Route-aware KV cache memory tiering compiler for LLM inference",
    author="Vamsi Sai Ranga Mangina",
    author_email="harsha.mangena@gmail.com",
    url="https://github.com/harsha-mangena/routekv-compiler",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "triton>=2.1.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "rich>=13.0.0",
        "datasets>=2.18.0",
        "einops>=0.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-cov",
            "black",
            "isort",
            "mypy",
        ],
        "viz": [
            "matplotlib>=3.8",
            "seaborn>=0.13",
            "plotly>=5.20",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

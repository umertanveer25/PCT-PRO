from setuptools import setup, find_packages

setup(
    name="pct_pro_engine",
    version="1.0.0",
    description="Phylogenetic Cognitive Transformer Pro: A State-of-the-Art IoMT-IDS Framework",
    author="Umer Tanveer, Yar Muhammad",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow",
        "xgboost",
        "scikit-learn",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research"
    ],
    python_requires='>=3.8',
)

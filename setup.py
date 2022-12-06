from setuptools import setup, find_packages

__version__ = "0.0.1"

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="TSB_UAD",
    version=__version__,
    description="Time Series Anomaly Detection Benchmark",
    classifiers=CLASSIFIERS,
    author="Teja",
    author_email="tejabogireddy19@gmail.com",
    packages=find_packages(),
    zip_safe=True,
    license="",
    url="https://github.com/TheDatumOrg/TSB-UAD",
    entry_points={},
    install_requires=[
        "arch==5.3.1",
        "hurst==0.0.5",
        "matplotlib==3.5.3",
        "numpy==1.21.6",
        "pandas==1.3.5",
        "scikit-learn==0.22",
        "scipy==1.7.3",
        "statsmodels==0.13.2",
        "tsfresh==0.8.1",
        "tslearn==0.4.1",
        "tensorboard==2.9.1",
        "tensorboard-data-server==0.6.1",
        "tensorboard-plugin-wit==1.8.1",
        "tensorflow==2.9.1",
        "tensorflow-estimator==2.9.0",
        "tensorflow-io-gcs-filesystem==0.26.0",
        "keras==2.9.0",
        "Keras-Preprocessing==1.1.2",
        "matrixprofile==1.1.10",
        "networkx==2.5.1",
        "stumpy==1.8.0",
        ]
)

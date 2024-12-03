from setuptools import setup, find_packages

__version__ = "0.0.3"

CLASSIFIERS = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Information Analysis",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="TSB_UAD",
    version=__version__,
    description="Time Series Anomaly Detection Benchmark",
    classifiers=CLASSIFIERS,
    author="The DATUM Lab",
    author_email="john@paparrizos.org",
    packages=find_packages(),
    zip_safe=True,
    license="Apache-2.0 license",
    url="https://github.com/TheDatumOrg/TSB-UAD",
    entry_points={},
    install_requires=[
        'numpy>=1.24.3',
        'matplotlib>=3.7.5',
        'pandas>=2.0.3',
        'arch>=5.3.1',
        'tsfresh>=0.20.2',
        'hurst>=0.0.5',
        'tslearn>=0.6.3',
        'cython>=3.0.10',
        'scikit-learn>=1.3.2',
        'stumpy>=1.12.0',
        'tensorflow>=2.13.0',
        'networkx>=3.1',
        ]
)

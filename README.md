# TSB-AUD: An End-to-End Anomaly Detection Benchmark Suite for Univariate Time-Series Data

TSB-UAD is a new open, end-to-end benchmark suite to ease the
evaluation of univariate time-series anomaly detection methods. Overall, TSB-UAD
contains 12686 time series with labeled anomalies spanning
different domains with high variability of anomaly types, ratios,
and sizes. Specifically, TSB-UAD includes 10 previously proposed
datasets containing 900 time series from real-world data science applications. 
Motivated by flaws in certain datasets and evaluation strategies in the literature,
we study anomaly types and data transformations to contribute
two collections of datasets. Specifically, we generate 958 time series
using a principled methodology for transforming 126 time-series
classification datasets into time series with labeled anomalies. In
addition, we present a set of data transformations with which we
introduce new anomalies in the public datasets, resulting in 10828
time series (92 datasets) with varying difficulty for anomaly detection.

## Publication

"TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection"

John Paparrizos, Yuhao Kang, Ruey S. Tsay, Themis Palpanas, and Michael J. Franklin

under submission/review

## Datasets

Due to limitations in the upload size on GitHub, we host the datasets at a different location:

Public: http://chaos.cs.uchicago.edu/tsb-uad/public.zip

Synthetic: http://chaos.cs.uchicago.edu/tsb-uad/synthetic.zip

Artificial: http://chaos.cs.uchicago.edu/tsb-uad/artificial.zip

The UCR classification datasets used to generate the Artificial datasets: http://chaos.cs.uchicago.edu/tsb-uad/UCR2018-NEW.zip

## Contributors

* John Paparrizos (University of Chicago)
* Yuhao Kang (University of Chicago)
* Alex Wu (University of Chicago)
* Teja Bogireddy (University of Chicago)

## Installation

This package works with Python 3.8 and pip 21. You need to have the following packages installed on the system.

```
pip install -r requirements.txt
```

Clone the repository

```
git clone https://github.com/yuhao12345/TSB-AD.git
```

Install locally

```
cd TSB-AD-main
pip install .
```

## Examples
* test_anomaly_detectors.ipynb : The performance of 10 popular anomaly detectors. 
* test_artificialConstruction.ipynb: The synthesized dataset based on anomaly construction. 
* test_transformer.ipynb: The effects of 11 transformations.
## Benchmark 
In ./data contains four folders: 

* benchmark/ contains ten public datasets. Below shows some typical outliers in these ten datasets.
<img width="1000" src="./doc/display_data.png"/>

* UCR2018-NEW/ contains 128 subfolders 

* artificial/ contains the data that are constructed based on UCR2018-NEW
<img width="500" src="./result/transform_subsequence.png"/>

* synthetic/ contains the data that are synthesized by local and global tranformations
<img width="500" src="./result/transform_global.png"/>
<img width="500" src="./result/transform_local.png"/>

## Anomaly Detector
We test ten algorithms in the module.

Below shows a result based on Autoencoder.

For each output figure, the left panel shows the real time series with outliers (red), anomaly score obtained by each anomaly detector, and the correpsonding TP/FP/TN/FN classification.

The right panel shows the ROC curve. AUC represents the area under the ROC curve. Larger AUC indicates better performance.

<img width="500" src="./result/AE.png"/>


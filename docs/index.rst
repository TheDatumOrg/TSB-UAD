.. TSB-UAD documentation master file, created by
   sphinx-quickstart on Sun Jun  9 18:10:29 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TSB-UAD's documentation!
===================================

.. toctree::
   :maxdepth: 1
   :hidden:

   overview/index
   algorithms/index
   evaluation/index
   data-processing/index
   



Overview
--------

TSB-UAD is a new open, end-to-end benchmark suite to ease the evaluation of univariate time-series anomaly detection methods. Overall, TSB-UAD contains 12686 time series with labeled anomalies spanning different domains with high variability of anomaly types, ratios, and sizes. Specifically, TSB-UAD includes 18 previously proposed datasets containing 1980 time series from real-world data science applications. Motivated by flaws in certain datasets and evaluation strategies in the literature, we study anomaly types and data transformations to contribute two collections of datasets. Specifically, we generate 958 time series using a principled methodology for transforming 126 time-series classification datasets into time series with labeled anomalies. In addition, we present a set of data transformations with which we introduce new anomalies in the public datasets, resulting in 10828 time series (92 datasets) with varying difficulty for anomaly detection.

1. `Real data <https://www.thedatum.org/datasets/TSB-UAD-Public.zip>`_
2. `Synthetic <https://www.thedatum.org/datasets/TSB-UAD-Synthetic.zip>`_
3. `Artificial <https://www.thedatum.org/datasets/TSB-UAD-Artificial.zip>`_


Installation
^^^^^^^^^^^^

Quick start:


TSB-UAD supports Python between 3.6 and 3.12. You can install it using:

.. code-block:: bash

   pip install TSB-UAD


Manual installation:

The following tools are required to install TSB-UAD from source:

- git
- conda (anaconda or miniconda)


Clone this `repository <https://github.com/TheDatumOrg/TSB-UAD.git>`_ using git and go into its root directory.

.. code-block:: bash

   git clone https://github.com/TheDatumOrg/TSB-UAD.git
   cd TSB-UAD/

Create and activate a conda-environment 'TSB'.

.. code-block:: bash

   conda env create --file environment.yml
   conda activate TSB

You can then install TSB-UAD with pip.

.. code-block:: bash

   pip install TSB-UAD

Usage
^^^^^

We depicts below a code snippet demonstrating how to use one anomaly detector (in this example, IForest).

.. code-block:: python

   import os
   import numpy as np
   import pandas as pd
   from TSB_UAD.models.iforest import IForest
   from TSB_UAD.models.feature import Window
   from TSB_UAD.utils.slidingWindows import find_length
   from TSB_UAD.vus.metrics import get_metrics

   df = pd.read_csv('data/benchmark/ECG/MBA_ECG805_data.out', header=None).to_numpy()
   data = df[:, 0].astype(float)
   label = df[:, 1]

   slidingWindow = find_length(data)
   X_data = Window(window = slidingWindow).convert(data).to_numpy()

   clf = IForest(n_jobs=1)
   clf.fit(X_data)
   score = clf.decision_scores_

   score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
   score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))


   results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
   for metric in results.keys():
       print(metric, ':', results[metric])

.. code-block:: bash

   AUC_ROC : 0.9216216369841076
   AUC_PR : 0.6608577550833885
   Precision : 0.7342093339374717
   Recall : 0.4010891089108911
   F : 0.5187770129662238
   Precision_at_k : 0.4010891089108911
   Rprecision : 0.7486112853253205
   Rrecall : 0.3097733542316151
   RF : 0.438214653167952
   R_AUC_ROC : 0.989123018780308
   R_AUC_PR : 0.9435238401582703
   VUS_ROC : 0.9734357459251715
   VUS_PR : 0.8858037295594041
   Affiliation_Precision : 0.9630674176380548
   Affiliation_Recall : 0.9809813654809071


License
^^^^^^^

The project is licensed under the `MIT license <https://mit-license.org>`_.

If you use TSB-UAD in your project or research, please cite the following papers:

   TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection
   John Paparrizos, Yuhao Kang, Paul Boniol, Ruey Tsay, Themis Palpanas, and Michael Franklin.
   Proceedings of the VLDB Endowment (PVLDB 2022) Journal, Volume 15, pages 1697–1711

   Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection
   John Paparrizos, Paul Boniol, Themis Palpanas, Ruey Tsay, Aaron Elmore, and Michael Franklin.
   Proceedings of the VLDB Endowment (PVLDB 2022) Journal, Volume 15, pages 2774‑2787

You can use the following BibTeX entries:

.. code-block:: bibtex

   @article{paparrizos2022tsb,
      title={Tsb-uad: an end-to-end benchmark suite for univariate time-series anomaly detection},
      author={Paparrizos, John and Kang, Yuhao and Boniol, Paul and Tsay, Ruey S and Palpanas, Themis and Franklin, Michael J},
      journal={Proceedings of the VLDB Endowment},
      volume={15},
      number={8},
      pages={1697--1711},
      year={2022},
      publisher={VLDB Endowment}
   }

.. code-block:: bibtex

   @article{paparrizos2022volume,
      title={{Volume Under the Surface: A New Accuracy Evaluation Measure for Time-Series Anomaly Detection}},
      author={Paparrizos, John and Boniol, Paul and Palpanas, Themis and Tsay, Ruey S and Elmore, Aaron and Franklin, Michael J},
      journal={Proceedings of the VLDB Endowment},
      volume={15},
      number={11},
      pages={2774--2787},
      year={2022},
      publisher={VLDB Endowment}
   }

Contributors
^^^^^^^^^^^^

- Paul Boniol (Inria, ENS)
- Qinghua Liu (Ohio State University)
- John Paparrizos (Ohio State University)
- Emmanouil Sylligardos (Inria, ENS)
- Ashwin Krishna (IIT Madras)
- Yuhao Kang (University of Chicago)
- Alex Wu (University of Chicago)
- Teja Bogireddy (University of Chicago)
- Themis Palpanas (Université Paris Cité)


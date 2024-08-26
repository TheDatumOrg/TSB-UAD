# In practice

In TSB-UAD, we provide a unique fonction to retrieve all evaluation measures.

```{eval-rst}  
.. autoclass:: TSB_UAD.vus.metrics.get_metrics
    :members:

```

## Example of usage

We depicts below the usage of get\_metrics.

```python
import os
import numpy as np
import pandas as pd
from TSB_UAD.utils.visualisation import plotFig
from TSB_UAD.models.sand import SAND
from TSB_UAD.models.feature import Window
from TSB_UAD.utils.slidingWindows import find_length
from TSB_UAD.vus.metrics import get_metrics

#Read data
filepath = 'PATH_TO_TSB_UAD/ECG/MBA_ECG805_data.out'
df = pd.read_csv(filepath, header=None).dropna().to_numpy()
name = filepath.split('/')[-1]

data = df[:,0].astype(float)
label = df[:,1].astype(int)

#Pre-processing
slidingWindow = find_length(data)

# Run SAND (offline)
modelName='SAND (offline)'
clf = SAND(pattern_length=slidingWindow,subsequence_length=4*(slidingWindow))
clf.fit(data,overlaping_rate=int(1.5*slidingWindow))
score = clf.decision_scores_

#Post-processing
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()


#Print accuracy
results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
for metric in results.keys():
    print(metric, ':', results[metric])
```
```
AUC_ROC : 0.996779310807228
AUC_PR : 0.8942079947725918
Precision : 0.7393483709273183
Recall : 0.9735973597359736
F : 0.8404558404558404
Precision_at_k : 0.9735973597359736
Rprecision : 0.7394705860012913
Rrecall : 0.9790057437116261
RF : 0.8425439890952773
R_AUC_ROC : 0.9996748675955897
R_AUC_PR : 0.9911647851406946
VUS_ROC : 0.9993050973645579
VUS_PR : 0.9802087454821152
Affiliation_Precision : 0.9825340283920497
Affiliation_Recall : 1.0
```

### References

* [Paparrizos et al. 2022] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay, Aaron Elmore, and Michael J. Franklin. 2022. Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. Proc. VLDB Endow. 15, 11 (July 2022), 2774–2787.

* [Tatbul et al. 2018] N. Tatbul, T.J. Lee, S. Zdonik, M. Alam, J. Gottschlich, in Advances in Neural Information Processing Systems, vol. 31

* [Huet et al. 2022] Alexis Huet, Jose Manuel Navarro, and Dario Rossi. 2022. Local Evaluation of Time Series Anomaly Detection Algorithms. In Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD '22).
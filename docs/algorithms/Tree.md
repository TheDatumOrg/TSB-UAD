![icon](../../images/method_icons/tree.png "icon")
# Tree-based methods


## Isolation Forest

Isolation Forest (IForest) is a density-based and the most famous tree-based approach for anomaly detection. IForest tries to isolate the outlier from the rest of the normal points of subsequences [Liu et al. 2008].
The key idea remains on the fact that, in a normal distribution, anomalies are more likely to be isolated (i.e., requiring fewer random partitions to be isolated) than normal instances. 
If we assume the latter statement, we only have to produce a partitioning process that indicates well the isolation degree (i.e., anomalous degree) of instances.

The TSB-kit implementation of IForest is a wrapper of [Scikit-learn implementation of IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).

```{eval-rst}  
.. autoclass:: tsb_kit.models.iforest.IForest
    :members:

```

### Example

Here is a code snippet that shows how to run Isolation Forest.

```python
import os
import numpy as np
import pandas as pd
from tsb_kit.utils.visualisation import plotFig
from tsb_kit.models.iforest import IForest
from tsb_kit.models.feature import Window
from tsb_kit.utils.slidingWindows import find_length
from tsb_kit.vus.metrics import get_metrics

#Read data
filepath = 'PATH_TO_TSB_UAD/ECG/MBA_ECG805_data.out'
df = pd.read_csv(filepath, header=None).dropna().to_numpy()
name = filepath.split('/')[-1]

data = df[:,0].astype(float)
label = df[:,1].astype(int)

#Pre-processing    
slidingWindow = find_length(data)
X_data = Window(window = slidingWindow).convert(data).to_numpy()


#Run IForest
modelName='IForest'
clf = IForest(n_jobs=1)
x = X_data
clf.fit(x)
score = clf.decision_scores_

# Post-processing
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))

#Plot result
plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName) 

#Print accuracy
results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
for metric in results.keys():
    print(metric, ':', results[metric])
```
```
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
```
![Result](../../images/method_results/IForest.png "Iforest Result")

### References

* [Liu et al. 2008] F. T. Liu, K. M. Ting, and Z.-H. Zhou. 2008. Isolation Forest. In Proceedings of the International Conference on Data Mining (ICDM), pp. 413–422. IEEE. ISBN 978-0-7695-3502-9. DOI:10.1109/ICDM.2008.17.

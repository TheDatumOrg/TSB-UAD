![icon](../../images/method_icons/graph.png "icon")
# Graph-based methods

## Series2Graph

Series2Graph [Boniol and Palpanas 2020] is a graph-based appraoch for time seires anomaly detection. It converts the time series into a directed graph with nodes representing the usual types of subsequences and edges representing the frequency of the transitions between types of subsequences. Moreover, an extension of Series2Graph proposed in the literature, named DADS [Schneider et al. 2021], proposes a distributed implementation and, therefore, a much more scalable method for large time series.


### Example

```python
import os
import numpy as np
import pandas as pd
from tsb_kit.utils.visualisation import plotFig
from tsb_kit.models.series2graph import Series2Graph
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

# Run Series2Graph
modelName='Series2Graph'
clf = Series2Graph(pattern_length=slidingWindow)
clf.fit(data)
clf.score(query_length=slidingWindow*2,dataset=data)
score = clf.decision_scores_

#Post-processing
score = np.array([score[0]]*math.ceil(query_length//2) + list(score) + [score[-1]]*(query_length//2))
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()


#Plot result
plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName)

#Print accuracy
results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
for metric in results.keys():
    print(metric, ':', results[metric])
```
```
AUC_ROC : 0.9791393752142049
AUC_PR : 0.8207550472208245
Precision : 0.7847222222222222
Recall : 0.7458745874587459
F : 0.7648054145516074
Precision_at_k : 0.7458745874587459
Rprecision : 0.4783287995269071
Rrecall : 0.796712220241632
RF : 0.5977696308872898
R_AUC_ROC : 0.9839017047884915
R_AUC_PR : 0.8408434964392455
VUS_ROC : 0.978472608857905
VUS_PR : 0.8203200675085416
Affiliation_Precision : 0.9282931033407741
Affiliation_Recall : 0.9989565215573869
```
![Result](../../images/method_results/Series2Graph.png "Series2Graph Result")

### References

* [Boniol and Palpanas 2020] Paul Boniol and Themis Palpanas. 2020. Series2Graph: graph-based subsequence anomaly detection for time series. Proc. VLDB Endow. 13, 12

* [Schneider et al. 2021] J. Schneider, P. Wenig, and T. Papenbrock. 2021. Distributed detection of sequential anomalies in univariate time series. The VLDB Journal, 30(4): 579–602
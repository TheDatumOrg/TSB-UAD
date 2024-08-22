![icon](../../images/method_icons/distrib.png "icon")
# Distribution-based methods

## Histogram-based Outlier Score (HBOS)

Histogram-based outlier detection (HBOS) [Goldstein et al. 2012] is an efficient unsupervised method. It assumes the feature independence and calculates the degree of outlyingness by building histograms. The methods is not dedicated for time series. Neverhteless, it can be used for point anomaly detection and subsequence as well, if we consider a subsequence as a vector.

```{eval-rst}  
.. autoclass:: TSB_UAD.models.hbos.HBOS
    :members:

```

```python
import os
import numpy as np
import pandas as pd
from TSB_UAD.utils.visualisation import plotFig
from TSB_UAD.models.hbos import HBOS
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
X_data = Window(window = slidingWindow).convert(data).to_numpy()


#Run HBOS
modelName='PCA'
clf = HBOS()
clf.fit(X_data)
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

### Example

### References

* [Goldstein et al. 2012] Goldstein, Markus and Andreas R. Dengel. “Histogram-based Outlier Score (HBOS): A fast Unsupervised Anomaly Detection Algorithm.” (2012).


## One-Class Support Vector Machine (OCSVM)

One-Class Support Vector Machine (OCSVM) is a typical distribution-based example, which aims to separate the instances from an origin and maximize the distance from the hyperplane separation [Schölkopf et al. 1999] or spherical separation [Tax and Duin 2004]. The anomalies are identified with points of high decision score, i.e., far away from the separation hyper-plane. This method is a variant of the classical Support Vector Machine for classification tasks [Hearst et al. 1998].

The TSB-kit implementation of OCSVM is a wrapper of [Scikit-learn implementation of OneClassSVM](https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html).


```{eval-rst}  
.. autoclass:: TSB_UAD.models.ocsvm.OCSVM
    :members:

```

### Example

```python
import os
import numpy as np
import pandas as pd
from TSB_UAD.utils.visualisation import plotFig
from TSB_UAD.models.ocsvm import OCSVM
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
data_train = data[:int(0.1*len(data))]
data_test = data

X_train = Window(window = slidingWindow).convert(data_train).to_numpy()
X_test = Window(window = slidingWindow).convert(data_test).to_numpy()

X_train_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_train.T).T
X_test_ = MinMaxScaler(feature_range=(0,1)).fit_transform(X_test.T).T


#Run OCSVM
modelName='OCSVM'
clf = OCSVM(nu=0.05)
clf.fit(X_train_, X_test_)
score = clf.decision_scores_

# Post-processing
score = np.array([score[0]]*math.ceil((slidingWindow-1)/2) + list(score) + [score[-1]]*((slidingWindow-1)//2))
score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()

#Plot result
plotFig(data, label, score, slidingWindow, fileName=name, modelName=modelName) 

#Print accuracy
results = get_metrics(score, label, metric="all", slidingWindow=slidingWindow)
for metric in results.keys():
    print(metric, ':', results[metric])
```
```
AUC_ROC : 0.9416967787322199
AUC_PR : 0.4592289027872978
Precision : 0.6402266288951841
Recall : 0.7458745874587459
F : 0.6890243902439025
Precision_at_k : 0.7458745874587459
Rprecision : 0.4007206588881263
Rrecall : 0.7967914438502675
RF : 0.5332568942659819
R_AUC_ROC : 0.9983442451461604
R_AUC_PR : 0.9119783238745204
VUS_ROC : 0.9905101824629529
VUS_PR : 0.8021253491270806
Affiliation_Precision : 0.9798093961448288
Affiliation_Recall : 0.9970749874410433
```
![Result](../../images/method_results/OCSVM.png "OCSVM Result")

### References

* [Schölkopf et al. 1999] B. Sch ̈olkopf, R. C. Williamson, A. Smola, J. Shawe-Taylor, and J. Platt. 1999. Support vector method for novelty detection. NeurIPS, 12.

* [Tax and Duin 2004] D. M. Tax and R. P. Duin. 2004. Support vector data description. Machine learning, 54(1): 45–66.

* [Hearst et al. 1998] M. A. Hearst, S. T. Dumais, E. Osuna, J. Platt, and B. Scholkopf. July 1998. Support vector machines. IEEE Intelligent Systems and their Applications, 13(4): 18–28. ISSN 1094-7167. DOI: 10.1109/5254.708428
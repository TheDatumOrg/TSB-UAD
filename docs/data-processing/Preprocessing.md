# Preprocessing Time Series

Even though all methods proposed in TSB-kit work for time series anomaly detection, not all methods require the same input shape. Some (like SAND and MatrixProfile) require the time series as a NumPy array of shape (n\_samples,) with n\_samples being the number of points in the time series, some others (like LOF or IForest) require a NumPy array of shape (n\_samples-length,m) corresponding to all consecutive subsequences of length m in the time series. Consequently, we provide code snippets below to show how to preprocess the time series.


## Reading the time series

The code snippet below shows how to read the time series and the labels from a TSB-UAD-formatted time series.

```python
import os
import numpy as np
import pandas as pd
from TSB_UAD.models.feature import Window


#Read data
filepath = 'PATH_TO_TSB_UAD/ECG/MBA_ECG805_data.out'
df = pd.read_csv(filepath, header=None).dropna().to_numpy()
name = filepath.split('/')[-1]

data = df[:,0].astype(float)
label = df[:,1].astype(int)
```

## Extracting all consecutive subsequences

You can use the Window class for methods that require a NumPy array of shape (n\_samples-length,m) corresponding to all consecutive subsequences of length m in the time series.

```{eval-rst}  
.. autoclass:: TSB_UAD.models.feature.Window
    :members:

```

### Example

```python
slidingWindow = 10 #User-defined subsequence length
X_data = Window(window = slidingWindow).convert(data).to_numpy()
```

## Computing automatically which subsequence length to use.

For the specific case of subsequence anomaly detection, most methods require a subsequence length as a parameter. This parameter can strongly influence the anomaly score and which subsequence will be considered an anomaly or not. 

In some use cases, the user knows exactly what length to use (e.g., special interest in abnormal days or hours in an electricity consumption time series). However, in many applications, such length might be difficult to guess in advance. 

In TSB-kit, we provide a function to automatically compute an "optimal" subsequence length. This function computes the optimal lag that maximizes auto-correlation and returns it. 

### Example

```python
import os
import numpy as np
import pandas as pd
from TSB_UAD.utils.slidingWindows import find_length

#Read data
filepath = 'PATH_TO_TSB_UAD/ECG/MBA_ECG805_data.out'
df = pd.read_csv(filepath, header=None).dropna().to_numpy()
name = filepath.split('/')[-1]

data = df[:,0].astype(float)
label = df[:,1].astype(int)

slidingWindow = find_length(data)
```

Please note that such a method is not perfect and is not guaranteed to return a relevant subsequence length. This is especially the case of non-periodic or non-cyclic times series. In some scenarios, it might be better to test multiple subsequence lengths instead of considering only the subsequence length provided by the function described above.   
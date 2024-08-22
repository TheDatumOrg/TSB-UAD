![VUS](../../images/measures/VUS.jpg "VUS measures")

# VUS-based Evaluation

In arder to limit the limitation dues to label misaligments, range-based extensions for ROC and PR curves have been introduced. THe latter introduce a new continuous label to enable more flexibility in measuring detected anomaly ranges. An other extension called Volume Under the Surface (VUS) for both ROC and PR curves have been proposed. VUS extends the mathematical model of Range-AUC measures by varying the buffer length. 

## Range-AUC-ROC and Range-AUC-PR

To compute the ROC curve and PR curve for a subsequence, we need to extend to definitions of TPR, FPR, and Precision. 

The first step is to add a buffer region at the boundary of outliers. The idea is that there should be a transition region between the normal and abnormal subsequences to accommodate the false tolerance of labeling in the ground truth (as discussed, this is unavoidable due to the mapping of discrete data to continuous time series). An extra benefit is that this buffer will give some credit to the high anomaly score in the vicinity of the outlier boundary, which is what we expected with the application of a sliding window originally. 

By default, the width of the buffer region at each side is half of the period $w$ of the time series (the period is an intrinsic characteristic of the time series). Differently, this parameter can be set into the average length of anomaly sizes or can be set to a desired value by the user.

The traditional binary label is extended to a continuous value. Formally, for a given buffer length $\ell$, the positions $s,e \in [0,|label|]$ the beginning and end indexes of a labeled anomaly (i.e., sections of continuous $1$ in $label$), we define the continuous $label_r$ as follows:

$\forall i \in [0,|label|],$

* ${label_ {\ell}}_ i = (1-\frac{|s-i|}{\ell})^{\frac{1}{2}} \text{ if: } s-\frac{\ell}{2} \leq i < s$

* ${label_ {\ell}}_ i = 1 \text{ if: } s \leq i < e$

* ${label_ {\ell}}_ i = (1-\frac{|e-i|}{\ell})^{\frac{1}{2}} \text{ if: } e \leq i < e+\frac{\ell}{2}$

* ${label_ {\ell}}_ i = 0 \text{ if: } i < s \text{ and } e < i$

When the buffer regions of two discontinuous outliers overlap, the label will be the superposition of these two orange curves with one as the maximum value. Using this new continuous label, one can compute $TP_\ell$, $FP_\ell$, $TN_\ell$ and $FN_\ell$ similarly as follows:


* $TP_ {\ell} = label_ {\ell}^\top \cdot pred$

* $FP_ {\ell} = (I- label_ {\ell})^\top \cdot pred$

* $TN_ {\ell} = (I- label_ {\ell})^\top \cdot (I-pred)$

* $FN_ {\ell} = label_ {\ell}^\top \cdot (I-pred)$

The total number of positive points P in this case naively should be $P_ {{\ell}_ 0} = TP_ {\ell}+ FN_ {\ell} = label_ {\ell}^\top \cdot I$. Here, we define it as:

* $P_ {\ell} = (label+label_ {\ell})^\top \cdot \frac{I}{2}$

* $N_ {\ell} = |label_ {\ell}|-P_ {\ell}$

The reason is twofold. When the length of the outlier is several periods, $P_ {{\ell}_ 0}$ and $P_ {\ell}$ are similar because the ratio of the buffer region to the whole anomaly region is small. When the length of the outlier is only half-period, the size of the buffer region is nearly two times the original abnormal region. In other words, to pursue false tolerance, the relative change we make to the ground truth is too significant. We use the average of $label$ and $label_ {\ell}$ to limit this change.

We finally generalize the point-based $Recall$, $Precision$, and $FPR$ to the range-based variants. Formally, following the definition of $R$ and $P$ as the set of anomalies range and detected predicted anomaly range, we define $TPR_ {\ell}$, $FPR_ {\ell}$, and $Precision_ {\ell}$:

* $TPR_ \ell=Recall_ {\ell}=\frac{TP_ {\ell}}{P_ {\ell}}.\sum_ {R_ i \in R} \frac{ExistenceR(R_ i,P)}{|R|}$

* $FPR_ {\ell}=\frac{FP_ {\ell}}{N_ {\ell}}$

* $Precision_ {\ell}=\frac{TP_ {\ell}}{TP_ {\ell}+FP_ {\ell}}$


Note that $TPR_ r=Recall_ r$. Moreover, for the recall computation, we incorporate the idea of Existence Reward [Tatbul et al. 2018], which is the ratio of the number of detected subsequence outliers to the total number of subsequence outliers. However, consistent with their work [Tatbul et al. 2018], we do not include the Existence ratio in the definition of range-precision. 

## VUS-ROC and VUS-PR

Range-AUC family of measures chooses the width of the buffer region to be half of a subsequence length $\ell$ of the time series. Such buffer length can be either set based on the knowledge of an expert (e.g., the usual size of arrhythmia in an electrocardiogram) or set automatically using the time series's period (which can easily be computed using either technique based on cross-correlation or the Fourier transform). Since the period is an intrinsic property of the time series, we can compare various algorithms on the same basis. However, a different approach may get a slightly different period. In addition, there are multi-period time series. So other groups may get different range-AUC because of the difference in the period. As a matter of fact, the parameter $\ell$, if not well set, can strongly influence range-AUC measures. To eliminate this influence, two generalizations of the range-AUC family of measures have been introduced.

The solution is to compute ROC and PR curves for different buffer lengths from 0 to the $\ell$. Therefore, the ROC and the PR curves become a surface in a three-dimensional space. Then, the overall accuracy measure corresponds to the Volume Under the Surface (VUS) for either the ROC surface (VUS-ROC) or PR surface (VUS-PR). As the R-AUC-ROC and R-AUC-PR are measures independent of the threshold on the anomaly score, the VUS-ROC and VUS-PR are independent of both the threshold and buffer length. Formally, given $Th=[Th_0,Th_1,...Th_N]$ with $0=Th_0<Th_1<...<Th_N=1$, and $\mathcal{L}=[\ell_0,\ell_1,...,\ell_L]$ with $0=\ell_0<\ell_1< ... < \ell_L = \ell$, we define VUS-ROC as:


$VUS\text{-}ROC = \frac{1}{4}\sum_ {w=1}^{L} \sum_ {k=1}^{N} \Delta^{(k,w)} . \Delta^{w}$

With:


* $\Delta^{(k,w)} = \Delta^{k}_ {TPR_ {\ell_ w}}.\Delta^{k}_ {FPR_ {\ell_ w}}+\Delta^{k}_ {TPR_ {\ell_ {w-1}}}.\Delta^{k}_ {FPR_ {\ell_ {w-1}}}$
* $\Delta^{k}_ {FPR_ {\ell_w}} = FPR_ {\ell_ w}(Th_ {k})-FPR_ {\ell_ w}(Th_ {k-1})$
* $\Delta^{k}_ {TPR_ {\ell_w}} = TPR_ {\ell_ w}(Th_ {k-1})+TPR_ {\ell_ w}(Th_ {k})$
* $\Delta^{w} = |\ell_ w - \ell_ {w-1}|$

Similarly, We can compute VUS-PR as follows:


$VUS\text{-}PR = \frac{1}{4}\sum_ {w=1}^{L} \sum_ {k=1}^{N} \Delta^{(k,w)} . \Delta^{w}$

With:

* $\Delta^{(k,w)} = \Delta^{k}_ {Pr_ {\ell_ w}}.\Delta^{k}_ {Re_ {\ell_ w}}+\Delta^{k}_ {Pr_ {\ell_ {w-1}}}.\Delta^{k}_ {Re_ {\ell_ {w-1}}}$
* $\Delta^{k}_ {Re_ {\ell_ w}} = Recall_ {\ell_ w}(Th_ {k})-Recall_ {\ell_ w}(Th_ {k-1})$
* $\Delta^{k}_ {Pr_ {\ell_ w}} = Precision_ {\ell_ w}(Th_ {k-1})+Precision_ {\ell_ w}(Th_ {k})$
* $\Delta^{w} = |\ell_ w - \ell_ {w-1}|$


From the above equations, we observe that the computation of VUS measures requires $O(N.L)$. In comparison, range-AUC measures require $O(N)$.
Thus, the application of VUS versus range-AUC depends on our knowledge of which buffer length to set. If one user knows which would be the most appropriate buffer length, range-AUC-based measures are preferable compared to VUS-based measures.
However, if there exists an uncertainty on $\ell$, then setting a range and using VUS increases the flexibility of the usage and the robustness of the evaluation. 

### References

* [Paparrizos et al. 2022] John Paparrizos, Paul Boniol, Themis Palpanas, Ruey S. Tsay, Aaron Elmore, and Michael J. Franklin. 2022. Volume under the surface: a new accuracy evaluation measure for time-series anomaly detection. Proc. VLDB Endow. 15, 11 (July 2022), 2774–2787.

* [Tatbul et al. 2018] N. Tatbul, T.J. Lee, S. Zdonik, M. Alam, J. Gottschlich, in Advances in Neural Information Processing Systems, vol. 31







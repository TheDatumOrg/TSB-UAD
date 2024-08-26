# Introduction

We define here the evaluation measures used in TSB-kit.
We first introduce formal notations. Then, we review in detail previously proposed evaluation measures for time-series anomaly detection methods. 
We review notations for the time series and anomaly score sequence.

## Time Series Notation

A time series $T \in \mathbb{R}^n$ is a sequence of
real-valued numbers $T_i\in\mathbb{R}, [T_1,T_2,...,T_n]$, where
$n=|T|$ is the length of $T$, and $T_i$ is the $i^{th}$ point of $T$. We
are typically interested in local regions of the time series, known as
subsequences. A subsequence $T_{i,\ell} \in \mathbb{R}^\ell$ of a time
series $T$ is a continuous subset of the values of $T$ of length $\ell$
starting at position $i$. Formally,
$T_{i,\ell} = [T_i, T_{i+1},...,T_{i+\ell-1}]$. 

## Anomaly Score Sequence

For a time series $T \in \mathbb{R}^n$, an AD method $A$
returns an anomaly score sequence $S_T$. For point-based approaches
(i.e., methods that return a score for each point of $T$), we have
$S_T \in \mathbb{R}^n$. For range-based approaches (i.e., methods that
return a score for each subsequence of a given length $\ell$), we have
$S_T \in \mathbb{R}^{n-\ell}$. Overall, for range-based (or
subsequence-based) approaches, we define $S_T = [S_{T,1},S_{T,2},...,S_{T,n-\ell}]$ with $S_{T,i} \in [0,1]$.
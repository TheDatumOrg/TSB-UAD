import numpy as np
from .utils.metrics import metricor


def generate_curve(label,score,slidingWindow, version='opt', thre=250):
    if version =='opt_mem':
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume_opt_mem(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
    else:
        tpr_3d, fpr_3d, prec_3d, window_3d, avg_auc_3d, avg_ap_3d = metricor().RangeAUC_volume_opt(labels_original=label, score=score, windowSize=slidingWindow, thre=thre)
        
    X = np.array(tpr_3d).reshape(1,-1).ravel()
    X_ap = np.array(tpr_3d)[:,:-1].reshape(1,-1).ravel()
    Y = np.array(fpr_3d).reshape(1,-1).ravel()
    W = np.array(prec_3d).reshape(1,-1).ravel()
    Z = np.repeat(window_3d, len(tpr_3d[0]))
    Z_ap = np.repeat(window_3d, len(tpr_3d[0])-1)
    
    return Y, Z, X, X_ap, W, Z_ap,avg_auc_3d, avg_ap_3d


def get_metrics(score, labels, metric='all', version='opt', slidingWindow=5, thre=250):
    """Compute all (or some) evaluation measures for a given score and labels.
    
    Parameters
    ----------
    score : numpy array of shape (n_samples,)
        The input score to evaluate.
    labels : numpy array of shape (n_samples,)
        the labels to compare the score with. It have to be composed of 0 and 1 (1 indicating if the point is an anomaly).
    slidingWindow: int, 
        Buffer length for Range-based AUC measures and for VUS-based measures. 
        For Range-AUC, the buffer length 
        is exactly equals to slidingWindow. For VUS-based measures, the buffer length varies from 0 to ``2*slidingWindow``.
    version : string, optional, default='opt'
        Implementation of VUS.

        - if 'opt', run the default implementation
        - if 'opt_mem', run the optimized implementation, but more complex in memory

    thre : int, optional, default=250
        Number of thresholds for VUS

    metric : string, optional, default='vus'
        compute a subset or all metrics:
        
        - if 'vus', compute and store in a dictionary the following measures (the string is the key of each measure). 

            - 'R_AUC_ROC', Range-adapted version of AUC-ROC [Paparrizos et al. 2022]
            - 'R_AUC_PR', Range-adapted version of AUC-PR [Paparrizos et al. 2022]
            - 'VUS_ROC', Volume under the surface for ROC [Paparrizos et al. 2022]
            - 'VUS_PR', Volume under the surface for PR [Paparrizos et al. 2022]

        - if 'all', compute all measures and store them in a dictionary (the string is the key of each measure). The threhold based measures are computed with a predifined threshold (``score_mu + 3*score_sigma``). In total here are the measures:

            - 'AUC_ROC', Area Under the ROC Curve
            - 'AUC_PR', Area Under the Precision-Recall Curve
            - 'Precision', generic Precision
            - 'Recall', generic Recall
            - 'F', generic F-score (with beta equals to 1)
            - 'Precision_at_k', generic precision at k.
            - 'Rprecision', Time series-adapted Precision [Tatbul et al. 2018].
            - 'Rrecall', Time series-adapted Recall [Tatbul et al. 2018].
            - 'RF', Time series-adapted F-score (with beta equals to 1) [Tatbul et al. 2018].
            - 'R_AUC_ROC', Range-adapted version of AUC-ROC [Paparrizos et al. 2022]
            - 'R_AUC_PR', Range-adapted version of AUC-PR [Paparrizos et al. 2022]
            - 'VUS_ROC', Volume under the surface for ROC [Paparrizos et al. 2022]
            - 'VUS_PR', Volume under the surface for PR [Paparrizos et al. 2022]
            - 'Affiliation_Precision', Affiliation-based precision [Huet et al. 2022]
            - 'Affiliation_Recall', Affiliation-based recall [Huet et al. 2022]
    
    Returns
    -------
    metrics : dictionary
        contains the accuracy values for all evaluation measures.
    """

    metrics = {}
    if metric == 'vus':
        grader = metricor()
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, slidingWindow, version, thre)

        metrics['VUS_ROC'] = VUS_ROC
        metrics['VUS_PR'] = VUS_PR

        return metrics

    elif metric == 'range_auc':
        grader = metricor()
        R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
        
        metrics['R_AUC_ROC'] = R_AUC_ROC
        metrics['R_AUC_PR'] = R_AUC_PR

        return metrics

    elif metric == 'auc':
        
        grader = metricor()
        AUC_ROC = grader.metric_new_auc(labels, score, plot_ROC=False)
        _, _, AUC_PR = grader.metric_PR(labels, score)

        metrics['AUC_ROC'] = AUC_ROC
        metrics['AUC_PR'] = AUC_PR

        return metrics
    
    else :
        from .basic_metrics import basic_metricor
        
        grader = metricor()
        _, _, _, _, _, _,VUS_ROC, VUS_PR = generate_curve(labels, score, slidingWindow, version, thre)
        R_AUC_ROC, R_AUC_PR, _, _, _ = grader.RangeAUC(labels=labels, score=score, window=slidingWindow, plot_ROC=True)
        grader = basic_metricor()
        AUC_ROC, Precision, Recall, F, Rrecall, ExistenceReward, OverlapReward, Rprecision, RF, Precision_at_k = grader.metric_new(labels, score, plot_ROC=False)
        _, _, AUC_PR = grader.metric_PR(labels, score)

        from .affiliation.generics import convert_vector_to_events
        from .affiliation.metrics import pr_from_events

        discrete_score = np.array(score > 0.5, dtype=np.float32)
        events_pred = convert_vector_to_events(discrete_score)
        events_gt = convert_vector_to_events(labels)
        Trange = (0, len(discrete_score))
        affiliation_metrics = pr_from_events(events_pred, events_gt, Trange)
        metrics['AUC_ROC'] = AUC_ROC
        metrics['AUC_PR'] = AUC_PR
        metrics['Precision'] = Precision
        metrics['Recall'] = Recall
        metrics['F'] = F
        metrics['Precision_at_k'] = Precision_at_k
        metrics['Rprecision'] = Rprecision
        metrics['Rrecall'] = Rrecall
        metrics['RF'] = RF
        metrics['R_AUC_ROC'] = R_AUC_ROC
        metrics['R_AUC_PR'] = R_AUC_PR
        metrics['VUS_ROC'] = VUS_ROC
        metrics['VUS_PR'] = VUS_PR
        metrics['Affiliation_Precision'] = affiliation_metrics['Affiliation_Precision']
        metrics['Affiliation_Recall'] = affiliation_metrics['Affiliation_Recall']
        
        if metric == 'all':
            return metrics
        else:
            return metrics[metric]

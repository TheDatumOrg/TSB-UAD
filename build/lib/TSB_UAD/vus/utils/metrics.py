from sklearn import metrics
import numpy as np
import math

class metricor:
    def __init__(self, a = 1, probability = True, bias = 'flat', ):
        self.a = a
        self.probability = probability
        self.bias = bias

    def detect_model(self, model, label, contamination = 0.1, window = 100, is_A = False, is_threshold = True):
        if is_threshold:
            score = self.scale_threshold(model.decision_scores_, model._mu, model._sigma)
        else:
            score = self.scale_contamination(model.decision_scores_, contamination = contamination)
        if is_A is False:
            scoreX = np.zeros(len(score)+window)
            scoreX[math.ceil(window/2): len(score)+window - math.floor(window/2)] = score 
        else:
            scoreX = score
        
        self.score_=scoreX
        L = self.metric(label, scoreX)
        return L

        
    def labels_conv(self, preds):
        '''return indices of predicted anomaly
        '''

        # p = np.zeros(len(preds))
        index = np.where(preds >= 0.5)
        return index[0]
    
    def labels_conv_binary(self, preds):
        '''return predicted label
        '''
        p = np.zeros(len(preds))
        index = np.where(preds >= 0.5)
        p[index[0]] = 1
        return p 

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start +AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue +=  bi
            if i in p:
                MyValue += bi
        return MyValue/MaxValue

    def Cardinality_factor(self, Anomolyrange, Prange):  ## Changed this is better reduced if conditions
        score = 0 
        start = Anomolyrange[0]
        end = Anomolyrange[1]
        print(start, end)
        for i in Prange:
            if not(i[1]<start or i[0]>end):
                score+=1
        if score == 0:
            return 0
        else:
            return 1/score

    def b(self, i, length):
        bias = self.bias 
        if bias == 'flat':
            return 1
        elif bias == 'front-end bias':
            return length - i + 1
        elif bias == 'back-end bias':
            return i
        else:
            if i <= length/2:
                return i
            else:
                return length - i + 1
            
    def scale_threshold(self, score, score_mu, score_sigma):
        return (score >= (score_mu + 3*score_sigma)).astype(int)

    def metric_new(self, label, score, plot_ROC=False, alpha=0.2,coeff=3):
        if np.sum(label) == 0:
            print('All labels are 0.')
            return None

        if np.isnan(score).any() or score is None:
            print('Score must not be none.')
            return None

        #auc-roc1
        auc = metrics.roc_auc_score(label, score)
        if plot_ROC:
            fpr, tpr, thresholds  = metrics.roc_curve(label, score)
            display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
            display.plot()            

        #precision, recall, F
        preds = score > (np.mean(score)+coeff*np.std(score))  
        if not(np.any(preds)):
            preds= score > (np.mean(score)+2*coeff*np.std(score)/3) 
            if np.any(preds):
                preds = score > (np.mean(score)+1*coeff*np.std(score)/3) 
                if np.any(preds):
                    preds = score > (np.mean(score) + 1*np.std(score))
    
        Precision, Recall, F, Support = metrics.precision_recall_fscore_support(label, preds, zero_division=0)
        precision = Precision[1]
        recall = Recall[1]
        f = F[1]

        #range based anomaly 
        Rrecall, ExistenceReward, OverlapReward = self.range_recall_new(label, preds, alpha)
        Rprecision = self.range_recall_new(preds, label, 0)[0]
        
        if Rprecision + Rrecall==0:
            Rf=0
        else:
            Rf = 2 * Rrecall * Rprecision / (Rprecision + Rrecall)
        
        # top-k
        k = int(np.sum(label))
        threshold = np.percentile(score, 100 * (1-k/len(label)))
        
        # precision_at_k = metrics.top_k_accuracy_score(label, score, k)
        p_at_k = np.where(preds > threshold)[0]
        TP_at_k = sum(label[p_at_k])
        precision_at_k = TP_at_k/k
        
        L = np.array([auc, precision, recall, f, Rrecall, ExistenceReward, OverlapReward, Rprecision, Rf, precision_at_k])
        if plot_ROC:
            return L, fpr, tpr
        return L

    def metric_new_auc(self, label, score, plot_ROC=False, alpha=0.2,coeff=3):
        if np.sum(label) == 0:
            print('All labels are 0. Label must have groud truth value for calculating AUC score.')
            return None

        if np.isnan(score).any() or score is None:
            print('Score must not be none.')
            return None

        #area under curve
        auc = metrics.roc_auc_score(label, score)
        
        return auc

    def metric_PR(self, label, score):
        precision, recall, thresholds = metrics.precision_recall_curve(label, score)
        # plt.figure()
        # disp = metrics.PrecisionRecallDisplay(precision=precision, recall=recall)
        # disp.plot()
        AP = metrics.auc(recall, precision)
        #AP = metrics.average_precision_score(label, score)
        return precision, recall, AP
        
    def range_recall_new(self, labels, preds, alpha):   


        p = np.where(preds == 1)[0]    # positions of predicted label==1
        range_pred = self.range_convers_new(preds)  
        range_label = self.range_convers_new(labels)
        
        Nr = len(range_label)    # total # of real anomaly segments

        ExistenceReward = self.existence_reward(range_label, p)


        OverlapReward = 0
        for i in range_label:
            OverlapReward += self.w(i, p) * self.Cardinality_factor(i, range_pred)


        score = alpha * ExistenceReward + (1-alpha) * OverlapReward
        if Nr != 0:
            return score/Nr, ExistenceReward/Nr, OverlapReward/Nr
        else:
            return 0,0,0


    def range_convers_new(self, label):
        '''
        input: arrays of binary values 
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        L = []
        i = 0
        j = 0 
        while j < len(label):
            # print(i)
            while label[i] == 0:
                i+=1
                if i >= len(label):  #?
                    break            #?
            j = i+1
            # print('j'+str(j))
            if j >= len(label):
                if j==len(label):
                    L.append((i,j-1))
    
                break
            while label[j] != 0:
                j+=1
                if j >= len(label):
                    L.append((i,j-1))
                    break
            if j >= len(label):
                break
            L.append((i, j-1))
            i = j
        return L
    
    def existence_reward(self, labels, preds):
        '''
        labels: list of ordered pair 
        preds predicted data
        '''

        score = 0
        for i in labels:
            if np.sum(np.multiply(preds <= i[1], preds >= i[0])) > 0:
                score += 1
        return score
    
    def num_nonzero_segments(self, x):
        count=0
        if x[0]>0:
            count+=1
        for i in range(1, len(x)):
            if x[i]>0 and x[i-1]==0:
                count+=1
        return count

    def extend_postive_range(self, x, window=5):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1] 
            
            
            x1 = np.arange(e+1,min(e+window//2+1,length))
            label[x1] += np.sqrt(1 - (x1-e)/(window))
            
            x2 = np.arange(max(s-window//2,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(window))
            
        label = np.minimum(np.ones(length), label)
        return label

    
    def extend_postive_range_individual(self, x, percentage=0.2):
        label = x.copy().astype(float)
        L = self.range_convers_new(label)   # index of non-zero segments
        length = len(label)
        for k in range(len(L)):
            s = L[k][0] 
            e = L[k][1] 
            
            l0 = int((e-s+1)*percentage)
            
            x1 = np.arange(e,min(e+l0,length))
            label[x1] += np.sqrt(1 - (x1-e)/(2*l0))
            
            x2 = np.arange(max(s-l0,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(2*l0))
            
        label = np.minimum(np.ones(length), label)
        return label
    
    def TPR_FPR_RangeAUC(self, labels, pred, P, L):
        product = labels * pred
        
        TP = np.sum(product)
        
        # recall = min(TP/P,1)
        P_new = (P+np.sum(labels))/2      # so TPR is neither large nor small
        # P_new = np.sum(labels)
        recall = min(TP/P_new,1)
        # recall = TP/np.sum(labels)
        # print('recall '+str(recall))
        
        
        existence = 0
        for seg in L:
            if np.sum(product[seg[0]:(seg[1]+1)])>0:
                existence += 1
                
        existence_ratio = existence/len(L)
        # print(existence_ratio)
        
        # TPR_RangeAUC = np.sqrt(recall*existence_ratio)
        # print(existence_ratio)
        TPR_RangeAUC = recall*existence_ratio
        
        FP = np.sum(pred) - TP
        # TN = np.sum((1-pred) * (1-labels))
        
        # FPR_RangeAUC = FP/(FP+TN)
        N_new = len(labels) - P_new
        FPR_RangeAUC = FP/N_new
        
        Precision_RangeAUC = TP/np.sum(pred)
        
        return TPR_RangeAUC, FPR_RangeAUC, Precision_RangeAUC
    
    def RangeAUC(self, labels, score, window=0, percentage=0, plot_ROC=False, AUC_type='window'):
        # AUC_type='window'/'percentage'
        score_sorted = -np.sort(-score)
        
        P = np.sum(labels)
        # print(np.sum(labels))
        if AUC_type=='window':
            labels = self.extend_postive_range(labels, window=window)
        else:   
            labels = self.extend_postive_range_individual(labels, percentage=percentage)
        
        # print(np.sum(labels))
        L = self.range_convers_new(labels)
        TF_list = np.zeros((252,2))
        Precision_list = np.ones(251)
        j=0
        for i in np.linspace(0, len(score)-1, 250).astype(int):
            threshold = score_sorted[i]
            # print('thre='+str(threshold))
            pred = score>= threshold
            TPR, FPR, Precision = self.TPR_FPR_RangeAUC(labels, pred, P,L)
            j+=1
            TF_list[j]=[TPR,FPR]
            Precision_list[j]=(Precision)
            
        TF_list[j+1]=[1,1]
                
        width = TF_list[1:,1] - TF_list[:-1,1]
        height = (TF_list[1:,0] + TF_list[:-1,0])/2
        AUC_range = np.dot(width,height)

        width_PR = TF_list[1:-1,0] - TF_list[:-2,0]
        height_PR = (Precision_list[1:] + Precision_list[:-1])/2
        AP_range = np.dot(width_PR,height_PR)
        
        if plot_ROC:
            return AUC_range, AP_range, TF_list[:,1], TF_list[:,0], Precision_list
        
        return AUC_range

    def new_sequence(self, label, sequence_original, window): 
        a = max(sequence_original[0][0] - window//2, 0)
        sequence_new = []
        for i in range (len(sequence_original) - 1):
            if sequence_original[i][1] + window//2 < sequence_original[i+1][0] - window//2:
                sequence_new.append((a, sequence_original[i][1] + window//2))
                a = sequence_original[i+1][0] - window//2
        sequence_new.append((a, min(sequence_original[len(sequence_original)-1][1] + window//2, len(label)-1)))
        return sequence_new

    def sequencing(self, x, L, window=5):
        label = x.copy().astype(float)
        length = len(label)
        
        for k in range(len(L)):
            s = L[k][0] 
            e = L[k][1] 
            
            x1 = np.arange(e+1,min(e+window//2+1,length))
            label[x1] += np.sqrt(1 - (x1-e)/(window))
            
            x2 = np.arange(max(s-window//2,0),s)
            label[x2] += np.sqrt(1 - (s-x2)/(window))
            
        label = np.minimum(np.ones(length), label)
        return label

    def RangeAUC_volume_opt(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize+1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)
        
        score_sorted = -np.sort(-score)

        tpr_3d=np.zeros((windowSize+1,thre+2))
        fpr_3d=np.zeros((windowSize+1,thre+2))
        prec_3d=np.zeros((windowSize+1,thre+1))

        auc_3d=np.zeros(windowSize+1)
        ap_3d=np.zeros(windowSize+1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)

        for k,i in enumerate(np.linspace(0, len(score)-1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score>= threshold
            N_pred[k]=np.sum(pred)

        for window in window_3d:

            labels = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels, seq, window)      

            TF_list = np.zeros((thre+2,2))
            Precision_list = np.ones(thre+1)
            j=0
            N_labels = 0

            for seg in l:
                N_labels += np.sum(labels[seg[0]:seg[1]+1])

            for i in np.linspace(0, len(score)-1, thre).astype(int):
                threshold = score_sorted[i]
                pred = score>= threshold
                
                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1]+1], pred[seg[0]:seg[1]+1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence = 0
                for seg in L:
                    if np.dot(labels[seg[0]:(seg[1]+1)],pred[seg[0]:(seg[1]+1)])>0:
                        existence += 1

                existence_ratio = existence/len(L)

                P_new = (P+N_labels)/2
                recall = min(TP/P_new,1)

                TPR = recall*existence_ratio
                N_new = len(labels) - P_new
                FPR = FP/N_new
                
                Precision = TP/N_pred[j]
                
                j+=1
                TF_list[j]=[TPR,FPR]
                Precision_list[j]=Precision
                
                
            TF_list[j+1]=[1,1]   # otherwise, range-AUC will stop earlier than (1,1)
            
            tpr_3d[window]=TF_list[:,0]
            fpr_3d[window]=TF_list[:,1]
            prec_3d[window]=Precision_list
            
            width = TF_list[1:,1] - TF_list[:-1,1]
            height = (TF_list[1:,0] + TF_list[:-1,0])/2
            AUC_range = np.dot(width,height)
            auc_3d[window]=(AUC_range)
            
            width_PR = TF_list[1:-1,0] - TF_list[:-2,0]
            height_PR = (Precision_list[1:] + Precision_list[:-1])/2

            AP_range = np.dot(width_PR,height_PR)
            ap_3d[window]=AP_range

        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d)/len(window_3d), sum(ap_3d)/len(window_3d)
    
    def RangeAUC_volume_opt_mem(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize+1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d=np.zeros((windowSize+1,thre+2))
        fpr_3d=np.zeros((windowSize+1,thre+2))
        prec_3d=np.zeros((windowSize+1,thre+1))

        auc_3d=np.zeros(windowSize+1)
        ap_3d=np.zeros(windowSize+1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        p = np.zeros((thre,len(score)))

        for k,i in enumerate(np.linspace(0, len(score)-1, thre).astype(int)):
            threshold = score_sorted[i]
            pred = score>= threshold
            p[k]=pred
            N_pred[k]=np.sum(pred)

        for window in window_3d:

            labels = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels, seq, window)

            TF_list = np.zeros((thre+2,2))
            Precision_list = np.ones(thre+1)
            j=0
            N_labels = 0

            for seg in l:
                N_labels += np.sum(labels[seg[0]:seg[1]+1])

            for i in np.linspace(0, len(score)-1, thre).astype(int):
                
                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1]+1], p[j][seg[0]:seg[1]+1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence = 0
                for seg in L:
                    if np.dot(labels[seg[0]:(seg[1]+1)],p[j][seg[0]:(seg[1]+1)])>0:
                        existence += 1

                existence_ratio = existence/len(L)

                P_new = (P+N_labels)/2
                recall = min(TP/P_new,1)

                TPR = recall*existence_ratio
                N_new = len(labels) - P_new
                FPR = FP/N_new
                
                Precision = TP/N_pred[j]
                j+=1
                
                TF_list[j]=[TPR,FPR]
                Precision_list[j]=Precision

            TF_list[j+1]=[1,1]
            

            tpr_3d[window]=TF_list[:,0]
            fpr_3d[window]=TF_list[:,1]
            prec_3d[window]=Precision_list
            
            width = TF_list[1:,1] - TF_list[:-1,1]
            height = (TF_list[1:,0] + TF_list[:-1,0])/2
            AUC_range = np.dot(width,height)
            auc_3d[window]=(AUC_range)
            
            width_PR = TF_list[1:-1,0] - TF_list[:-2,0]
            height_PR = (Precision_list[1:] + Precision_list[:-1])/2
            AP_range = np.dot(width_PR,height_PR)
            ap_3d[window]=(AP_range)

        return tpr_3d, fpr_3d, prec_3d, window_3d, sum(auc_3d)/len(window_3d), sum(ap_3d)/len(window_3d)
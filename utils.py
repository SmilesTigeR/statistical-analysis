import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix as c_matrix

def plot_roc(y, prob):
    fpr, tpr, _ = roc_curve(y, prob)
    c = auc(fpr, tpr)
    plt.xlim(0, 1)
    plt.plot([0, 1], [0, 1], linestyle = '--')
    plt.plot(fpr, tpr, label = 'AUC: {0:.3f}'.format(c))
    plt.legend(loc = 'lower right')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve')
    return c

def plot_ks(y, prob):
    fpr, tpr, threshold = roc_curve(y, prob)
    diff = tpr - fpr
    ks = np.max(diff)
    max_threshold = threshold[np.argmax(diff)]
    plt.xlim(0, 1)
    plt.plot(threshold, 1 - fpr, label = '1-FPR')
    plt.plot(threshold, 1 - tpr, label = '1-TPR')
    plt.axvline(x = max_threshold, color = 'black', ymin = (1-tpr)[np.argmax(diff)] + 0.03, ymax = (1-fpr)[np.argmax(diff)] - 0.03,
                linestyle = '--', label = 'KS Statistic: {0:.3f} at {1:.3f}'.format(ks, max_threshold))
    plt.legend(loc = 'lower right')
    plt.xlabel('Threshold')
    plt.ylabel('Percentage')
    plt.title('KS Chart')
    return (ks, max_threshold)
    
def confusion_matrix(y, prob, threshold):
    pred = [1 if p >= threshold else 0 for p in prob]
    cm = c_matrix(y, pred)
    cm = pd.DataFrame(cm)
    tn = cm.iloc[0, 0]
    tp = cm.iloc[1, 1]
    s = cm.sum()
    st = cm.sum(axis = 1)
    stotal = s.sum()
    
    cm['Predictive Value'] = [tn, tp]
    cm['Predictive Value'] = cm['Predictive Value']/st
    
    tmp = pd.DataFrame([[tn, tp, (tp+tn)/stotal]], columns = cm.columns, index = ['Specificity & Sensitivity'])
    tmp.iloc[:2, :2] = tmp.iloc[:2, :2].div(s)
    cm = pd.concat([cm, tmp])
    
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    return cm

def roc(y, prob, fpr_original):
    fpr, tpr, _ = roc_curve(y, prob, drop_intermediate = False)
    fpr = fpr[1:]
    tpr = tpr[1:]
    tpr = [1 if np.sum(fpr >= x) == 0 else tpr[np.where(fpr == np.min(fpr[fpr >= x]))[0][-1]] for x in fpr_original]
    return tpr

def EmpiricalBootstrappingROC(df, X, y, model, B, typ = 'bca', alpha = 0.05):
    bias = []
    
    def __resample_tpr__(df_resample):
        model.fit(df_resample[X], df_resample[y])
        prob = model.predict_proba(df_resample[X])[:, 1]
        tpr = roc(df_resample[y], prob, fpr_original)
        return tpr
    
    
    df = df.reset_index(drop = True)
    n = len(df)
    model.fit(df[X], df[y])
    prob = model.predict_proba(df[X])[:, 1]
    fpr_original, tpr_original, _ = roc_curve(df[y], prob)
    
    fpr_original = fpr_original[1:]
    tpr_original = tpr_original[1:]
    s = auc(fpr_original, tpr_original)
    
    if typ == 'bca':
        tprs = []
        for idx, _ in LeaveOneOut().split(df):
            df_resample = df.iloc[idx]
            tpr = __resample_tpr__(df_resample)
            tprs.append(tpr)
        tprs = np.array(tprs)
        m = np.mean(tprs, axis = 0)
        num = np.sum((tprs - m)**3, axis = 0)
        den = 6*(np.sum((tprs - m)**2, axis = 0))**(3/2)
        a = num/den
    
    tprs = []
    for i in range(B):
        df_resample = df.sample(n = n, replace = True)
        tprs.append(__resample_tpr__(df_resample))
    tprs = np.array(tprs)
    
    if typ == 'bca':
        z0 = 1/B*np.sum(tprs < tpr_original, axis = 0)
        z0 = np.apply_along_axis(norm.cdf, 0, z0)

        a1 = np.repeat(norm.ppf(alpha/2), len(tpr_original))
        a1 = z0 + (z0+a1)/(1-a*(z0+a1))
        a1 = np.apply_along_axis(norm.cdf, 0, a1)
        a1[np.isnan(a1)] = alpha/2
        tpr_lower = [np.quantile(tprs[:, x], a1[x]) for x in range(len(tpr_original))]
        a2 = np.repeat(norm.ppf(1-alpha/2), len(tpr_original))
        a2 = z0 + (z0+a2)/(1-a*(z0+a2))
        a2 = np.apply_along_axis(norm.cdf, 0, a2)
        a2[np.isnan(a2)] = 1-alpha/2
        tpr_upper = [np.quantile(tprs[:, x], a2[x]) for x in range(len(tpr_original))]
        
    elif typ == 'basic':
        tprs = -tprs + 2*tpr_original
        tpr_lower = [np.quantile(tprs[:, x], alpha/2) for x in range(len(tpr_original))]
        tpr_upper = [np.quantile(tprs[:, x], 1-alpha/2) for x in range(len(tpr_original))]

    fpr_original = [0] + list(fpr_original)
    tpr_original = [0] + list(tpr_original)
        
    tpr_lower = [0] + tpr_lower
    tpr_upper = [0] + tpr_upper
    
    plt.plot(fpr_original, tpr_original, color = 'orange', label = 'ROC: {0:.3f}'.format(s))
    plt.plot(fpr_original, tpr_lower, color = 'blue', label = '{0:.1f}% Lower Confidence Limit of ROC: {1:.3f}'.format((1-alpha)*100, auc(fpr_original, tpr_lower)))
    plt.plot(fpr_original, tpr_upper, color = 'blue', label = '{0:.1f}% Upper Confidence Limit of ROC: {1:.3f}'.format((1-alpha)*100, auc(fpr_original, tpr_upper)))
    plt.fill_between(fpr_original, tpr_lower, tpr_upper, color = 'cyan')
    plt.plot([0, 1], [0, 1], linestyle = '--')
    
    plt.legend(loc = 'lower right')
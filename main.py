import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from collections import Counter
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
import seaborn as sns; sns.set(style="ticks", color_codes=True)
import pickle
from imblearn.under_sampling import (RandomUnderSampler, 
                                     ClusterCentroids,
                                     TomekLinks,
                                     NeighbourhoodCleaningRule,
                                     NearMiss)
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
    # Model Package
from sklearn.neighbors import KNeighborsClassifier
    # Hyperparameter tuning
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.calibration import calibration_curve

    # Performance metrics
from sklearn.metrics import  confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score ,f1_score, roc_auc_score, auc ,plot_roc_curve
from sklearn.metrics import roc_curve
from scipy import interp

df = pd.read_excel('/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Academic Desk/2024Instability/model_instability/dataset/cnx_external_dataset.xlsx')

list_of_col = ['age','sex','stat','HT','lipid','BMI','waistcir','calfcir','exhaustion']

#df['edu'] =np.where(df['education']>=4 , 1,0)
#df['frail'] =np.where(df['frailscore']>=3 , 1,0)
#df['stat'] =np.where(df['status']>=4 , 1,0)

#transformer = make_column_transformer(MinMaxScaler(),['age'])
pd.options.display.max_columns = df.shape[1]
df = df.dropna(axis = 0, how = 'any')
df.info()
df.describe()
#list_col = df.columns.tolist()[1:36]
#print(list_col)
#df2.info()

       ## X Feature Selection 
X = df[list_of_col]
#X = transformer.fit_transform(X)
X.info()
X.hist()


    ## Y Selection
y = df[['frail']]
y.info()
y.hist()

# 2.1 Option "Dealing with unbalance data SMOTE: The synthetic minority oversampling technique"
sm = SMOTE(random_state=42)
X_sm, Y_sm = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(Y_sm))

    # Undersampling 
cc = RandomUnderSampler(random_state=42)
X_cc, Y_cc = cc.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(Y_cc))

# 2.2 Choosing data for learning (sm or cc or normal)
X = np.array(X_sm)
y = np.array(Y_sm)

#X = np.array(X_cc)
#Y = np.array(Y_cc)

#X = np.array(X)
#Y = np.array(Y)

## Model Parameter setting [best]

#best_param = _GS.best_params_
best_param = {'C': 2, 'class_weight': 'balanced', 'penalty': 'l2', 'solver': 'liblinear'}
print(best_param)
logista_GS = LogisticRegression(**best_param)

#KNN_GS.fit(X,y.ravel())
#print(KNN_GS)

cv = StratifiedKFold(n_splits=10)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots()
for i, (train, test) in enumerate(cv.split(X,y)):
        logista_GS.fit(X[train], y[train].ravel())
        viz = plot_roc_curve(logista_GS, X[test], y[test],
                                name='ROC fold {}'.format(i),
                                alpha=0.5, lw=1, ax=ax)
        interp_tpr = interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
ci_auc = sms.DescrStatsW(aucs).tconfint_mean()
ax.plot(mean_fpr, mean_tpr, color='b',
        label='Mean ROC,[95% CI]'+': %0.2f, [%0.2f - %0.2f]' % (mean_auc, ci_auc[0], ci_auc[1]),
        lw=2, alpha=.8)

ci_tpr = sms.DescrStatsW(tprs).tconfint_mean()
tprs_upper = np.minimum(ci_tpr[1], 1)
tprs_lower = np.maximum(ci_tpr[0], 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label= '95% CI')
ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
title="Receiver Operating Characteristic of Logistic Regression using Rebalanced Data by SMOTE")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()

ci_tpr = sms.DescrStatsW(tprs).tconfint_mean()
tprs_upper = np.minimum(ci_tpr[1], 1)
tprs_lower = np.maximum(ci_tpr[0], 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label= '95% CI')
ax.set_xlabel("1 - Specificity")
ax.set_ylabel("Sensitivity")
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
title="Receiver Operating Characteristic of Logistic Regression using Rebalanced Data by SMOTE")
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()


y_pred = cross_val_predict(logista_GS, X, y,  cv =10 )
y_pred

print(y_pred)

MLP_M = confusion_matrix(y,y_pred)

print('Confusion Matrix : \n', MLP_M)

total_M = sum(sum(MLP_M))

#####from confusion matrix calculate accuracy
accuracy_M =(MLP_M[0,0]+MLP_M[1,1])/total_M
print ('Accuracy : ', accuracy_M)

specificity_M = MLP_M[0,0]/(MLP_M[0,0]+MLP_M[0,1])
print('Specificity : ', specificity_M )

sensitivity_M = MLP_M[1,1]/(MLP_M[1,0]+MLP_M[1,1])
print('Sensitivity : ', sensitivity_M)

# precision score and recall(sensitivity)
print('precision_score',precision_score(y, y_pred))
print('recall_score',recall_score(y, y_pred))
print("f1_score:" , f1_score(y, y_pred))
print("AUC:" , roc_auc_score(y, y_pred))

## 95%CI##
print("Original ROC area: {:0.3f}".format(roc_auc_score(y, y_pred)))

n_bootstraps = 1000
rng_seed = 42  # control reproducibility
bootstrapped_scores = []

rng = np.random.RandomState(rng_seed)
for i in range(n_bootstraps):
    # bootstrap by sampling with replacement on the prediction indices
    indices = rng.randint(0, len(y_pred), len(y_pred))
    if len(np.unique(y[indices])) < 2:
        # We need at least one positive and one negative sample for ROC AUC
        # to be defined: reject the sample
        continue

    score = roc_auc_score(y[indices], y_pred[indices])
    bootstrapped_scores.append(score)
    print("Bootstrap #{} ROC area: {:0.3f}".format(i + 1, score))
        
plt.hist(bootstrapped_scores, bins=50)
plt.title('Histogram of the bootstrapped ROC AUC scores')
plt.show()          

sorted_scores = np.array(bootstrapped_scores)
sorted_scores.sort()

# Computing the lower and upper bound of the 90% confidence interval
# You can change the bounds percentiles to 0.025 and 0.975 to get
# a 95% confidence interval instead.
confidence_lower = sorted_scores[int(0.05 * len(sorted_scores))]
confidence_upper = sorted_scores[int(0.95 * len(sorted_scores))]
print("Confidence interval for the score: [{:0.3f} - {:0.3}]".format(confidence_lower, confidence_upper))
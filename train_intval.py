"""
Created Date:       22/01/2025
Last update:        22/01/2025
Program:            Logistic Regression Modeling
@author:            Natthanaphop Isaradech, MD.
Description:        -
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split, StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay, roc_auc_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import os

# Define paths and configuration
df_path = '/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Academic Desk/2024Instability/model_instability/dataset/lp_internal_dataset.xlsx'
figure = '/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Academic Desk/2024Instability/model_instability/figures'
os.makedirs(figure, exist_ok=True)



def load_data(file_path, features, target_column):
    df = pd.read_excel(file_path).iloc[:, 1:]
    X = df[features]
    y = df[target_column]
    return X, y, df

def explore(X, y):
    """Exploratory Data Analysis"""
    print("Feature Info:")
    X.info()
    plt.figure(figsize=(10, 8))
    X.hist(bins=15, edgecolor='black', grid=False)
    plt.suptitle('Feature Histograms', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(figure, 'feature_histograms.png'))
    plt.show()

    print("Target Info:")
    y.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Target Variable Distribution', fontsize=16)
    plt.xlabel('Target Value', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(figure, 'target_distribution.png'))
    plt.show()
    # # Additional details about categorical features
    # print("Sex: 1 = Male, 2 = Female")
    # print("Status: 0 = Not living alone, 1 = Living alone")


def sampling(X, y, mode='over'):
    """Handles data sampling: oversampling or undersampling."""
    if mode == 'over':
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        print('Resampled dataset shape:', Counter(y_res))
    elif mode == 'under':
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)
        print('Resampled dataset shape:', Counter(y_res))
    else:
        X_res, y_res = X, y

    return np.array(X_res), np.array(y_res)

def logitML(X, y):
    """Performs logistic regression modeling with hyperparameter tuning."""
    model = LogisticRegression()

    param_grid = {
        'penalty': ['l1'],
        'C': np.arange(0.01, 100, 10),
        'class_weight': ['balanced', None],
        'solver': ['liblinear', 'saga']
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=10,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)
    return grid_search.best_estimator_, grid_search.best_params_

def roc_curve(model, model_name, X, y):
    """
    ROC curve function for cross-validation.
    Args:
        model: Trained model with best parameters.
        model_name: Name of the model.
    """
    cv = StratifiedKFold(n_splits=10)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        model.fit(X[train], y[train].ravel())
        viz = RocCurveDisplay.from_estimator(model, X[test], y[test],
                                             name=f'ROC fold {i}',
                                             alpha=0.5, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    ci_auc = sm.stats.DescrStatsW(aucs).tconfint_mean()
    ax.plot(mean_fpr, mean_tpr, color='black',
            label=f'Mean ROC [95% CI]: {mean_auc:.2f} [{ci_auc[0]:.2f} - {ci_auc[1]:.2f}]',
            lw=2, alpha=.8)

    tprs_upper = np.minimum(np.percentile(tprs, 97.5, axis=0), 1)
    tprs_lower = np.maximum(np.percentile(tprs, 2.5, axis=0), 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                    label='95% CI')

    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_title(f"Receiver Operating Characteristic {model_name} using Rebalanced Data by SMOTE")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    file_path = os.path.join(figure, f"roc_curve_{model_name}.jpg")
    fig.savefig(file_path, format='jpg', bbox_inches='tight', pad_inches=0.1)
    print(f"ROC curve saved to {file_path}")
    plt.show()

# Load data
# Define features and target
features = ['age', 'sex', 'HT', 'lipid', 'BMI', 'waistcir', 'calfcir', 'exhaustion']
key = 'frail'

X, y, df = load_data(df_path, features, key)

# Explore data
explore(X, y)

# Resample data
X_res, y_res = sampling(X, y, mode='over')

# Train the model
best_model, best_params = logitML(X_res, y_res)
print("Best Parameters:", best_params)
print("Best Model:", best_model)

# Plot ROC curve
roc_curve(best_model, "Logistic Regression", X_res, y_res)

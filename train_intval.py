"""
Created Date:       22/01/2025
Last update:        22/01/2025
Program:            Logistic Regression Modeling
@author:            Natthanaphop Isaradech, M.D.
Description:        -
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, cross_val_predict, train_test_split, StratifiedKFold, cross_validate
)
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from statsmodels.stats.weightstats import DescrStatsW
import statsmodels.api as sm
import seaborn as sns

def load_data(file_path, features, target_column, mode):
    df = pd.read_csv(file_path) #.iloc[:, 1:]
    if mode == 'sim':
        df = df.groupby(target_column).apply(lambda x: x.sample(frac=0.05, random_state=42))
    else:
        df = df
    X = df[features]
    y = df[target_column]
    return X, y, df

def explore(X, y, name):
    """Exploratory Data Analysis"""
    print("Feature Info:")
    X.info()
    plt.figure(figsize=(10, 8))
    X.hist(bins=15, edgecolor='black', grid=False)
    plt.suptitle('Feature Histograms', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(results, f'{name}feature_histograms.png'))
    plt.show()

    print("Target Info:")
    y.value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Target Variable Distribution', fontsize=16)
    plt.xlabel('Target Value', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(results, f'{name}target_distribution.png'))
    plt.show()
    # # Additional details about categorical features
    # print("Sex: 1 = Male, 2 = Female")
    # print("Status: 0 = Not living alone, 1 = Living alone")


def sampling(X, y, mode=None):
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
    model = LogisticRegression(random_state=48)

    param_grid = {
        'penalty': ['l1','l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'class_weight': ['balanced'],
        'solver': ["newton-cholesky", "sag", "saga", "lbfgs"],
        'max_iter': [500]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=10,
        n_jobs=-1
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
    
    file_path = os.path.join(results, f"roc_curve_{model_name}.jpg")
    fig.savefig(file_path, format='jpg', bbox_inches='tight', pad_inches=0.1)
    print(f"ROC curve saved to {file_path}")
    plt.show()
    
def bootstrap_metrics(model, X, y, model_name, n_iterations=1000):
    """
    Perform bootstrapping to calculate metrics with mean and 95% confidence intervals.

    Parameters:
    X (array-like): Feature dataset.
    y (array-like): True labels.
    model: Trained machine learning model.
    n_iterations (int): Number of bootstrap iterations.

    Returns:
    dict: Mean and 95% CI for each metric.
    """

    metrics = []
    n_size = len(X)

    for i in range(n_iterations):
        # Resample with replacement
        X_resampled, y_resampled = resample(X, y, replace=True, n_samples=n_size, random_state=i)

        # Predict on resampled data
        y_pred = model.predict(X_resampled)
        y_prob = model.predict_proba(X_resampled)[:, 1]  # For AUROC

        # Compute metrics
        acc = accuracy_score(y_resampled, y_pred)
        sens = recall_score(y_resampled, y_pred)  # Sensitivity = Recall
        prec = precision_score(y_resampled, y_pred)
        f1 = f1_score(y_resampled, y_pred)
        auroc = roc_auc_score(y_resampled, y_prob)

        # Confusion matrix for specificity, PPV, and NPV
        tn, fp, fn, tp = confusion_matrix(y_resampled, y_pred).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0

        metrics.append((acc, sens, spec, prec, ppv, npv, f1, auroc))

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame(
        metrics,
        columns=["Accuracy", "Sensitivity", "Specificity", "Precision", "PPV", "NPV", "F1_Score", "AUROC"]
    )
    
    # Calculate mean and 95% confidence intervals
    results_model = {}
    for metric in metrics_df.columns:
        mean = metrics_df[metric].mean()
        std = np.std(metrics, ddof=1)  # Sample standard deviation
        se = std / np.sqrt(len(metrics))  # Standard error

        # Calculate 95% Confidence Interval
        z = stats.norm.ppf(0.975)  # z-score for 95% confidence
        ci_lower = mean - z * se
        ci_upper = mean + z * se
        results_model[metric] = f"{mean:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f})"

    # Plot histogram of AUROC with mean line and density curve
    plt.figure(figsize=(10, 6))
    sns.histplot(
        metrics_df['AUROC'], 
        bins=30, 
        kde=True, 
        color='blue', 
        alpha=0.7, 
        edgecolor='black',
        linewidth=1.2
    )
    
    # Add mean line
    auroc_mean = metrics_df['AUROC'].mean()
    plt.axvline(auroc_mean, color='red', linestyle='--', linewidth=1.5, label=f"Mean = {auroc_mean:.3f}")

    # Add legend
    plt.legend(loc='upper left')

    # Add title and labels
    plt.title("Histogram and Density of AUROC (Bootstrapping)", fontsize=14)
    plt.xlabel("AUROC", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(axis='y', alpha=0.3)
    
    
    if not os.path.exists(results):
        os.makedirs(results)
    plt.savefig(os.path.join(results, f"Histogram_of_AUROC_{model_name}.png"), bbox_inches='tight')

    print(f"Plot saved to {os.path.join(results, f'Histogram_of_AUROC_{model_name}.png')}")
    plt.show()
    
    y_pred = model.predict(X)
    disp = ConfusionMatrixDisplay.from_predictions(y, y_pred, cmap='Blues')  # Add optional styling
    plt.title("Confusion Matrix")  # Optional title
    plt.savefig(os.path.join(results, f"confusion_matrix_{model_name}.png"), bbox_inches='tight')  # Save the figure
    plt.show()
    # disp = ConfusionMatrixDisplay(confusion_matrix=model)
    # disp.plot()
    with open(os.path.join(results,f"result_{model_name}.json"), "w") as file:
        json.dump(results_model, file, indent=4)  # `indent=4` makes the file human-readable

    return results_model
  
# Load data
## Define paths and configuration
df_path = '/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Academic Desk/2024Instability/model_instability/dataset/GUSTO/gusto_dataset(Sheet1).csv'
results = '/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Academic Desk/2024Instability/model_instability/results/dev_val/full'
os.makedirs(results, exist_ok=True)

## Define features and target
features = ['age', 'sex', 'hyp', 'htn', 'hrt', 'ste', 'pmi', 'sysbp']
key = 'day30'

# X, y, df = load_data(df_path, features, key, mode = 'sim')
df_full = pd.read_csv(df_path)
df_full['sex'] = df_full['sex'].apply(lambda x: 1 if x == 'male' else 0)
df_full['pmi'] = df_full['pmi'].apply(lambda x: 1 if x == 'yes' else 0)

X = df_full[features]
y = df_full[key]

# Explore data
explore(X, y, name = 'Full')

# Resample data
X, y = sampling(X, y)

# Train the model
model_name = "FullLogit"
best_model, best_params = logitML(X, y)
print("Best Parameters:", best_params)
with open(os.path.join(results,f"best_params_{model_name}.json"), "w") as file:
        json.dump(best_params, file, indent=4)  # `indent=4` makes the file human-readable

# Plot ROC curve
# roc_curve(best_model, "Logistic Regression", X, y)
bootstrap_metrics(best_model, X, y, model_name)

# Sampling
results = '/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Academic Desk/2024Instability/model_instability/results/dev_val/reduced'
os.makedirs(results, exist_ok=True)
df_sam = df_full.groupby(key).apply(lambda x: x.sample(frac=0.025, random_state=42)).reset_index(drop=True)
X = df_sam[features]
y = df_sam[key]

# Train the model
model_name = "SamLogit"
best_model, best_params = logitML(X, y)
print("Best Parameters:", best_params)
with open(os.path.join(results,f"best_params_{model_name}.json"), "w") as file:
        json.dump(best_params, file, indent=4)  # `indent=4` makes the file human-readable

# Plot ROC curve
# roc_curve(best_model, "Logistic Regression", X, y)
bootstrap_metrics(best_model, X, y, model_name)
import os
import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from collections import Counter
from statsmodels.stats.weightstats import DescrStatsW

# Scikit-learn modules
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import (
    GridSearchCV, cross_val_score, cross_val_predict, train_test_split, 
    StratifiedKFold, cross_validate
)
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, 
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.calibration import calibration_curve

# Imbalanced-learn modules
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load dataset
def load_data(file_path, features, target_column):
    df = pd.read_excel(file_path) #.iloc[:, 1:]
    X = df[features]
    y = df[target_column]
    return X, y, df

# Perform exploratory data analysis (EDA)
def exploratory_data_analysis(df):
    print("Basic Information:")
    print(df.info())
    print("\nSummary Statistics:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # print("\nCorrelation Matrix:")
    # print(df.corr())

    # Plot histograms for all numerical columns
    df.hist(bins=15, figsize=(15, 10))
    plt.suptitle("Histograms of Numerical Features")
    plt.show()
    
def logitML(X, y, param_grid):
    """Performs logistic regression modeling with hyperparameter tuning."""
    model = LogisticRegression(random_state=48)

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
    plt.savefig(os.path.join(results, f"Histogram_of_BOOT_VAL_{model_name}.png"), bbox_inches='tight')

    print(f"Plot saved to {os.path.join(results, f'Histogram_of_BOOT_VAL_{model_name}.png')}")
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

# Train logistic regression with GridSearchCV
def train_model(X, y, param_grid, cv=10):
    logistic_model = LogisticRegression()
    grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)
    grid_search.fit(X, y)
    best_params = grid_search.best_params_
    best_model = LogisticRegression(**best_params)
    best_model.fit(X, y)
    return best_model

# Perform bootstrapping
def bootstrap_training(X, df, features, target_column, param_grid, n_bootstrap):
    bootstrap_models = []
    bootstrap_probs = []

    for i in range(n_bootstrap):
        # Resample dataset
        boot_df = resample(df, n_samples=len(df), random_state=i)
        X_boot = boot_df[features]
        y_boot = boot_df[target_column]

        # Train model on bootstrapped dataset
        best_model = train_model(X_boot, y_boot, param_grid, cv=10)
        bootstrap_models.append(best_model)
        bootstrap_probs.append(best_model.predict_proba(X)[:, 1])  # Predict on the original dataset

    np.save(results + "/bootstrap_probs.npy",  np.array(bootstrap_probs))
    return bootstrap_models, np.array(bootstrap_probs)

# Plot comparison of predicted probabilities
def plot_probability_comparison(original_probs, bootstrap_probs, lowess_2_5, lowess_97_5, n_bootstrap):
    plt.figure(figsize=(10, 6))

    # Scatter plots for each bootstrapped model
    for i, probs in enumerate(bootstrap_probs):
        plt.scatter(original_probs, probs, 
                    color='grey',  
                    alpha=1,     # Set transparency
                    s=0.05)

    # Add ideal line for reference
    plt.plot([0, 1], [0, 1], 'k-', lw=1, label="Ideal Line")

    # Add LOWESS smoothed percentile lines
    plt.plot(lowess_2_5[:, 0], lowess_2_5[:, 1], 'k--', lw=1, label="LOWESS 2.5th Percentile Line")
    plt.plot(lowess_97_5[:, 0], lowess_97_5[:, 1], 'k--', lw=1, label="LOWESS 97.5th Percentile Line")

    # Plot settings
    plt.xlabel("Original Model Predicted Probabilities")
    plt.ylabel("Bootstrapped Models Predicted Probabilities")
    plt.title(f"Comparison of Predicted Probabilities: Original Model vs. {n_bootstrap} Bootstrapped Models")
    plt.grid(False)
    plt.legend()
    plt.savefig(results + '/probability_comparison.png')
    plt.show()

# Plot calibration curve for original and bootstrapped models
def plot_calibration_with_bootstrap(origin_predict, bootstrap_probs, y, n_bootstrap, n_bins=30):
    plt.figure(figsize=(10, 6))
    # Bootstrap models calibration curves
    for index, i in enumerate(bootstrap_probs):
        y_observed, y_prob = calibration_curve(y, i, n_bins=n_bins, strategy='uniform')
        plt.plot(y_observed, y_prob, color='grey', alpha=0.5, lw=1, label="Bootstrap Models" if index == 0 else "")  # Label only the first curve

    # Original model calibration curve
    y_observed, y_prob = calibration_curve(y, origin_predict, n_bins=n_bins, strategy='uniform')
    plt.plot(y_observed, y_prob, 'k--', lw=2, label="Original Model (Dashed Line)")

    
    # Ideal calibration line
    plt.plot([0, 1], [0, 1], 'k-', lw=1.5, label="Ideal Calibration Line")

    # Plot settings
    plt.xlabel("Observed Predicted Probability")
    plt.ylabel("Model Predicted Probability")
    plt.title(f"Model Calibration Instability with {n_bootstrap} Bootstrapped Models")
    plt.legend(loc='upper left', frameon=True, fontsize='small')
    plt.grid(True)
    plt.savefig(results + '/calibration_comparison.png')
    plt.show()

# Calculate LOWESS smoothed percentiles
def calculate_lowess_percentiles(bootstrap_probs, original_probs):
    percentile_2_5 = np.percentile(bootstrap_probs, 2.5, axis=0)
    percentile_97_5 = np.percentile(bootstrap_probs, 97.5, axis=0)
    lowess_2_5 = sm.nonparametric.lowess(percentile_2_5, original_probs, frac=0.3)
    lowess_97_5 = sm.nonparametric.lowess(percentile_97_5, original_probs, frac=0.3)
    return lowess_2_5, lowess_97_5

# Plot MAPE instability 
def plot_mape_instability(origin_predict, bootstrap_probs):
    # Transpose bootstrap_probs to match the shape for broadcasting
    bootstrap_probs = bootstrap_probs.T
    absolute_errors = np.abs(bootstrap_probs - origin_predict[:, np.newaxis])
    # Calculate Mean Absolute Prediction Error (MAPE)
    mape = np.mean(absolute_errors, axis = 1) * 100

    y_values = mape.flatten()
    # Repeat origin_predict values for each column in bootstrap_probs
    # x_values = np.repeat(origin_predict, absolute_errors.shape[1])
    x_values = origin_predict
    # Plot MAPE instability
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.5, s=0.3)
    # plt.axhline(mean_mape, color='red', linestyle='--', label=f"Mean MAPE: {mean_mape:.2f}%")
    plt.xlabel("Original Model: Predicted Probability")
    plt.ylabel("MAPE (%)")
    plt.ylim(0, 20)
    plt.title("MAPE Instability Plot")
    plt.grid(True)
    plt.legend()
    plt.savefig(results + '/mape.png')
    plt.show()
    # print(f"Mean MAPE: {mean_mape:.2f}%")

# Load dataset
## Define bootstraps and model training configuration
param_grid = {
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ["newton-cholesky", "sag", "saga", "lbfgs"],
        'max_iter': [1000]
    }
n_bootstrap = 500

# FULL DATASET
df_path = '/home/natthanaphop.isa/model_instability/dataset/gusto_dataset(Sheet1).csv'
df = pd.read_csv(df_path)
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['pmi'] = df['pmi'].apply(lambda x: 1 if x == 'yes' else 0)
## Define features and target
features = ['age', 'sex', 'hyp', 'htn', 'hrt', 'ste', 'pmi', 'sysbp']
key = 'day30'
X = df[features]
y = df[key]

## dev and internal validation
model_name = "FullLogit"
results = '/home/natthanaphop.isa/model_instability/results/dev_val/full'
os.makedirs(results, exist_ok=True)
best_model, best_params = logitML(X, y, param_grid)
print("Best Parameters:", best_params)
with open(os.path.join(results,f"best_params_{model_name}.json"), "w") as file:
        json.dump(best_params, file, indent=4)  # `indent=4` makes the file human-readable

# Plot ROC curve
# roc_curve(best_model, "Logistic Regression", X, y)
bootstrap_metrics(best_model, X, y, model_name)

## Instability
results = '/home/natthanaphop.isa/model_instability/results/instability/full'
os.makedirs(results, exist_ok=True)
## Train original model
origin_predict = best_model.predict_proba(X)[:, 1]
np.save(results + "/full_origin_predict.npy",  np.array(origin_predict))

# ## Bootstrap training
bootstrap_models, bootstrap_probs = bootstrap_training(X, df, features, key, param_grid, n_bootstrap)
# ## Calculate LOWESS smoothed percentiles
lowess_2_5, lowess_97_5 = calculate_lowess_percentiles(bootstrap_probs, origin_predict)

## Plot results
plot_mape_instability(origin_predict, bootstrap_probs)
plot_probability_comparison(origin_predict, bootstrap_probs, lowess_2_5, lowess_97_5, n_bootstrap)
plot_calibration_with_bootstrap(origin_predict, bootstrap_models, y, n_bootstrap)

# REDUCED DATASET
results = '/home/natthanaphop.isa/model_instability/results/dev_val/reduced'
os.makedirs(results, exist_ok=True)
df_sam = df.groupby(key).apply(lambda x: x.sample(frac=0.025, random_state=42)).reset_index(drop=True)
X = df_sam[features]
y = df_sam[key]

# Train the model
model_name = "SamLogit"
best_model, best_params = logitML(X, y, param_grid)
print("Best Parameters:", best_params)
with open(os.path.join(results,f"best_params_{model_name}.json"), "w") as file:
        json.dump(best_params, file, indent=4)  # `indent=4` makes the file human-readable

# Plot ROC curve
# roc_curve(best_model, "Logistic Regression", X, y)
bootstrap_metrics(best_model, X, y, model_name)

# SAMPLED DATASET
## Results
results = '/home/natthanaphop.isa/model_instability/results/instability/reduced'
os.makedirs(results, exist_ok=True)

# Train original model
origin_predict = best_model.predict_proba(X)[:, 1]
np.save(results + "/reduced_origin_predict.npy",  np.array(origin_predict))

# Bootstrap training
bootstrap_models, bootstrap_probs = bootstrap_training(X, df, features, key, param_grid, n_bootstrap)

# Calculate LOWESS smoothed percentiles
lowess_2_5, lowess_97_5 = calculate_lowess_percentiles(bootstrap_probs, origin_predict)

def plot_mape_instability2(origin_predict, bootstrap_probs):
    # Transpose bootstrap_probs to match the shape for broadcasting
    bootstrap_probs = bootstrap_probs.T
    absolute_errors = np.abs(bootstrap_probs - origin_predict[:, np.newaxis])
    # Calculate Mean Absolute Prediction Error (MAPE)
    mape = np.mean(absolute_errors, axis = 1) * 100

    y_values = mape.flatten()
    # Repeat origin_predict values for each column in bootstrap_probs
    # x_values = np.repeat(origin_predict, absolute_errors.shape[1])
    x_values = origin_predict
    
    # Plot MAPE instability
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=1, s=1)
    # plt.axhline(mean_mape, color='red', linestyle='--', label=f"Mean MAPE: {mean_mape:.2f}%")
    plt.xlabel("Original Model: Predicted Probability")
    plt.ylabel("MAPE (%)")
    plt.ylim(0, 20)
    plt.title("MAPE Instability Plot")
    plt.grid(True)
    plt.legend()
    plt.savefig(results + '/mape.png')
    plt.show()
    # print(f"Mean MAPE: {mean_mape:.2f}%")
    
# Plot results
plot_mape_instability2(origin_predict, bootstrap_probs)
plot_probability_comparison(origin_predict, bootstrap_probs, lowess_2_5, lowess_97_5, n_bootstrap)
plot_calibration_with_bootstrap(origin_predict, bootstrap_models, y, n_bootstrap)



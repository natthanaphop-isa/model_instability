import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load dataset
def load_data(file_path, features, target_column):
    df = pd.read_excel(file_path)
    X = df[features]
    y = df[target_column]
    return X, y, df

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
def bootstrap_training(X, y, df, features, target_column, param_grid, n_bootstrap):
    bootstrap_models = []
    bootstrap_probs = []

    for i in range(n_bootstrap):
        # Resample dataset
        boot_df = resample(df, n_samples=len(df), random_state=i)
        X_boot = boot_df[features]
        y_boot = boot_df[target_column]

        # Train model on bootstrapped dataset
        best_model = train_model(X_boot, y_boot, param_grid, cv=5)
        bootstrap_models.append(best_model)
        bootstrap_probs.append(best_model.predict_proba(X)[:, 1])  # Predict on the original dataset

    return bootstrap_models, np.array(bootstrap_probs)


# Plot comparison of predicted probabilities
def plot_probability_comparison(original_probs, bootstrap_probs, lowess_2_5, lowess_97_5, n_bootstrap):
    plt.figure(figsize=(10, 6))

    # Scatter plots for each bootstrapped model
    for i, probs in enumerate(bootstrap_probs):
        plt.scatter(original_probs, probs, alpha=0.6, s=2)

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
    plt.show()

# Plot calibration curve for original and bootstrapped models
def plot_calibration_with_bootstrap(original_model, bootstrap_models, X, y, n_bins=5):
    plt.figure(figsize=(10, 6))

    # Original model calibration curve
    original_probs = original_model.predict_proba(X)[:, 1]
    mean_predicted_prob, observed_fraction = calibration_curve(y, original_probs, n_bins=n_bins, strategy='uniform')
    plt.plot(mean_predicted_prob, observed_fraction, 'k--', lw=2, label="Original Model (Dashed Line)")

    # Bootstrap models calibration curves
    for i, model in enumerate(bootstrap_models):
        bootstrap_probs = model.predict_proba(X)[:, 1]
        mean_predicted_prob, observed_fraction = calibration_curve(y, bootstrap_probs, n_bins=n_bins, strategy='uniform')
        plt.plot(mean_predicted_prob, observed_fraction, color='grey', alpha=0.6, lw=1, label="Bootstrap Models" if i == 0 else "")  # Label only the first curve

    # Ideal calibration line
    plt.plot([0, 1], [0, 1], 'k-', lw=1.5, label="Ideal Calibration Line")

    # Plot settings
    plt.xlabel("Predicted Probability")
    plt.ylabel("Observed Predicted Probability")
    plt.title("Model Calibration with Bootstrap Instability")
    plt.legend(loc='upper left', frameon=True, fontsize='small')
    plt.grid(True)
    plt.show()


# Calculate LOWESS smoothed percentiles
def calculate_lowess_percentiles(bootstrap_probs, original_probs):
    percentile_2_5 = np.percentile(bootstrap_probs, 2.5, axis=0)
    percentile_97_5 = np.percentile(bootstrap_probs, 97.5, axis=0)
    lowess_2_5 = sm.nonparametric.lowess(percentile_2_5, original_probs, frac=0.3)
    lowess_97_5 = sm.nonparametric.lowess(percentile_97_5, original_probs, frac=0.3)
    return lowess_2_5, lowess_97_5

# Main workflow
file_path = '/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Academic Desk/2024Instability/model_instability/dataset/cnx_external_dataset.xlsx'
list_of_col = ['age', 'sex', 'HT', 'lipid', 'BMI', 'waistcir', 'calfcir', 'exhaustion']
target_column = 'frail'
param_grid = { 
    'penalty': ['l1', 'l2'],  
    'C': np.arange(1, 2, 0.5), 
    'class_weight': ['balanced', None], 
    'solver': ['liblinear', 'saga']  
}
n_bootstrap = 20

# Load dataset
X, y, df = load_data(file_path, list_of_col, target_column)

# Train original model
original_model = train_model(X, y, param_grid)

# Bootstrap training
bootstrap_models, bootstrap_probs = bootstrap_training(X, y, df, list_of_col, target_column, param_grid, n_bootstrap)

# Calculate LOWESS smoothed percentiles
lowess_2_5, lowess_97_5 = calculate_lowess_percentiles(bootstrap_probs, original_model.predict_proba(X)[:, 1])

# Plot results
plot_probability_comparison(original_model.predict_proba(X)[:, 1], bootstrap_probs, lowess_2_5, lowess_97_5, n_bootstrap)
plot_calibration_with_bootstrap(original_model, bootstrap_models, X, y)

# Main workflow
file_path = '/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Academic Desk/2024Instability/model_instability/dataset/lp_internal_dataset.xlsx'
list_of_col = ['age', 'sex', 'HT', 'lipid', 'BMI', 'waistcir', 'calfcir', 'exhaustion']
target_column = 'frail'
param_grid = { 
    'penalty': ['l1', 'l2'],  
    'C': np.arange(1, 2, 0.5), 
    'class_weight': ['balanced', None], 
    'solver': ['liblinear', 'saga']  
}
n_bootstrap = 20

# Load dataset
X, y, df = load_data(file_path, list_of_col, target_column)

# Train original model
original_model = train_model(X, y, param_grid)

# Bootstrap training
bootstrap_models, bootstrap_probs = bootstrap_training(X, y, df, list_of_col, target_column, param_grid, n_bootstrap)

# Calculate LOWESS smoothed percentiles
lowess_2_5, lowess_97_5 = calculate_lowess_percentiles(bootstrap_probs, original_model.predict_proba(X)[:, 1])

# Plot results
plot_probability_comparison(original_model.predict_proba(X)[:, 1], bootstrap_probs, lowess_2_5, lowess_97_5, n_bootstrap)
plot_calibration_with_bootstrap(original_model, bootstrap_models, X, y)

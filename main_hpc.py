import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os

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
    predictions = pd.DataFrame({'origin_predict':origin_predict})

    for i in range(n_bootstrap):
        # Resample dataset
        boot_df = resample(df, n_samples=len(df), random_state=i)
        X_boot = boot_df[features]
        y_boot = boot_df[target_column]

        # Train model on bootstrapped dataset
        best_model = train_model(X_boot, y_boot, param_grid, cv=10)
        bootstrap_models.append(best_model)
        bootstrap_probs.append(best_model.predict_proba(X)[:, 1])
        probs = pd.DataFrame({f'{i}_bootstrap_probs':best_model.predict_proba(X)[:, 1]})
        predictions = pd.concat([predictions, probs], axis=1)
        
    np.save(results + "/bootstrap_probs.npy",  np.array(bootstrap_probs))
    return bootstrap_models, np.array(bootstrap_probs), predictions

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
def plot_calibration_with_bootstrap(origin_predict, bootstrap_models, X, y, n_bootstrap, n_bins=5):
    plt.figure(figsize=(10, 6))

    # Original model calibration curve
    mean_predicted_prob, observed_fraction = calibration_curve(y, origin_predict, n_bins=n_bins, strategy='uniform')
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
    plt.scatter(origin_predict, y_values, alpha=0.5, s=0.3)
    # plt.axhline(mean_mape, color='red', linestyle='--', label=f"Mean MAPE: {mean_mape:.2f}%")
    plt.xlabel("Original Model: Predicted Probability")
    plt.ylabel("MAPE (%)")
    plt.ylim(0, 100)
    plt.xlim(0,1)
    plt.title("MAPE Instability Plot")
    plt.grid(True)
    plt.legend()
    plt.savefig(results + '/mape.png')
    plt.show()
    # print(f"Mean MAPE: {mean_mape:.2f}%")

# Load dataset
## Define bootstraps and model training configuration
param_grid = param_grid = {
        'penalty': ['l2'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'solver': ["newton-cholesky", "sag", "saga", "lbfgs"],
        'max_iter': [1000]
    }
n_bootstrap = 300

# FULL DATASET
## Results
df_path = '/home/natthanaphop.isa/model_instability/dataset/gusto_dataset(Sheet1).csv'
results = '/home/natthanaphop.isa/model_instability/results/instability/full'
os.makedirs(results, exist_ok=True)
# X, y, df = load_data(df_path, features, key, mode = 'sim')
df = pd.read_csv(df_path)
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['pmi'] = df['pmi'].apply(lambda x: 1 if x == 'yes' else 0)
## Define features and target
features = ['age', 'sex', 'hyp', 'htn', 'hrt', 'ste', 'pmi', 'sysbp']
key = 'day30'
X = df[features]
y = df[key]

## Train original model
original_model = train_model(X, y, param_grid)
origin_predict= original_model.predict_proba(X)[:, 1]

# ## Bootstrap training
bootstrap_models, bootstrap_probs, predictions = bootstrap_training(X, df, features, key, param_grid, n_bootstrap)
predictions.to_csv(results + '/full_predictions.csv')
# ## Calculate LOWESS smoothed percentiles
lowess_2_5, lowess_97_5 = calculate_lowess_percentiles(bootstrap_probs, origin_predict)

# ## Plot results
plot_mape_instability(origin_predict, bootstrap_probs)
plot_probability_comparison(origin_predict, bootstrap_probs, lowess_2_5, lowess_97_5, n_bootstrap)
plot_calibration_with_bootstrap(origin_predict, bootstrap_models, X, y, n_bootstrap)

# SAMPLED DATASET
## Results
df_path = '/home/natthanaphop.isa/model_instability/dataset/gusto_dataset(Sheet1).csv'
results = '/home/natthanaphop.isa/model_instability/results/instability/reduced'
os.makedirs(results, exist_ok=True)

# X, y, df = load_data(df_path, features, key, mode = 'sim')
df = pd.read_csv(df_path)
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['pmi'] = df['pmi'].apply(lambda x: 1 if x == 'yes' else 0)

df_sam = df.groupby(key).apply(lambda x: x.sample(frac=0.025, random_state=42)).reset_index(drop=True)
df = df_sam
X = df[features]
y = df[key]

# Train original model
original_model = train_model(X, y, param_grid)
origin_predict= original_model.predict_proba(X)[:, 1]

# Bootstrap training
bootstrap_models, bootstrap_probs, predictions = bootstrap_training(X, df, features, key, param_grid, n_bootstrap)
predictions.to_csv(results + '/small_predictions.csv')

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
    plt.xlim(0, 1)
    plt.ylim(0, 100)
    plt.title("MAPE Instability Plot")
    plt.grid(True)
    plt.legend()
    plt.savefig(results + '/mape.png')
    plt.show()
    # print(f"Mean MAPE: {mean_mape:.2f}%")
    
# Plot results
plot_mape_instability2(origin_predict, bootstrap_probs)
plot_probability_comparison(origin_predict, bootstrap_probs, lowess_2_5, lowess_97_5, n_bootstrap)
plot_calibration_with_bootstrap(origin_predict, bootstrap_models, X, y, n_bootstrap)
# Standard library
import json
import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.matplotlib_cache')
import statistics
from collections import Counter

# Third-party
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from alive_progress import alive_bar
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from sklearn.neural_network import MLPClassifier

# Load dataset
def load_data(file_path, features, target_column):
    df = pd.read_excel(file_path) #.iloc[:, 1:]
    X = df[features]
    y = df[target_column]
    return X, y, df

def sampling(X, y, mode=None):
    """Handles data sampling: oversampling or undersampling."""
    if mode == 'over':
        sm = SMOTE(random_state=30)
        X_res, y_res = sm.fit_resample(X, y)
        print('Resampled dataset shape:', Counter(y_res))
    elif mode == 'under':
        rus = RandomUnderSampler(random_state=30)
        X_res, y_res = rus.fit_resample(X, y)
        print('Resampled dataset shape:', Counter(y_res))
    else:
        X_res, y_res = X, y

    return np.array(X_res), np.array(y_res)

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

# Train MLP with GridSearchCV
def train_model(X, y, param_grid, cv=10):
    mlp_model = MLPClassifier(random_state=RANDOM_STATE, max_iter=500)
    grid_search = GridSearchCV(estimator=mlp_model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=0)
    grid_search.fit(X, y)
    best_model = mlp_model.set_params(**grid_search.best_params_)
    return best_model

def c_statistic_optimism(model, n_bootstrap, x, y):
    """
    Calculates C-statistic optimism using bootstrap samples.

    Args:
        model (sklearn model): Trained model.
        n_bootstrap (int): Number of bootstrap samples.
        x (pd.DataFrame): Feature data.
        y (pd.Series): Target data.

    Returns:
         list: List of c-statistics from each bootstrap sample.
    """
    data = pd.concat([x, y], axis=1)
    c_stat_bs = []

    with alive_bar(n_bootstrap, title="C-statistic Bootstrap") as bar:
        for _ in range(n_bootstrap):
            data_sample = data.sample(n=len(data), replace=True).reset_index(drop=True)
            x_sample, y_sample = data_sample.iloc[:, :-1], data_sample.iloc[:, -1]
            model.fit(x_sample, y_sample)
            c = roc_auc_score(y_sample, model.predict_proba(x_sample)[:, 1])
            c_stat_bs.append(c)
            bar()
    return c_stat_bs

def plot_histogram_with_density(data, figure_path, model_name="", title="C-statistics stability"):
    """
    Plots a histogram with a density plot and a reference line.

    Args:
        data (list): Data to plot.
        title (str): Title of the plot.
    """
    mean = statistics.mean(data)
    ci = sms.DescrStatsW(data).tconfint_mean(alpha=0.05, alternative='two-sided')
    minimum = min(data)
    maximum = max(data)
    plt.figure(figsize=(8, 6))
    sns.histplot(data, kde=True, bins=50, color='skyblue')
    plt.axvline(mean, color='red', linestyle='--', linewidth=1, label=f'Apparent AuROC: {mean:.3f}')
    plt.axvline(ci[0], color='black', linestyle='--', linewidth=1, label=f'95% LCI: {ci[0]:.3f}')
    plt.axvline(ci[1], color='black', linestyle='--', linewidth=1, label=f'95% UCI: {ci[1]:.3f}')
    plt.axvline(minimum, color='grey', linestyle='--', linewidth=1, label=f'Minimum: {minimum:.3f}')
    plt.axvline(maximum, color='grey', linestyle='--', linewidth=1, label=f'Maximum: {maximum:.3f}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f"{model_name} — {title}" if model_name else title)
    plt.legend()
    plt.savefig(figure_path + '/optimism.png')
    plt.show()

# Perform bootstrapping
def bootstrap_training(model, X, df, features, target_column, param_grid, n_bootstrap, origin_predict, result_path):
    bootstrap_models = []
    bootstrap_probs = []
    predictions = pd.DataFrame({'origin_predict':origin_predict})

    with alive_bar(n_bootstrap, title="Prediction Stability Bootstrap") as bar:
        for i in range(n_bootstrap):
            # Resample dataset
            # boot_df = resample(df, replace = True, n_samples=len(df))
            boot_df = df.sample(n=len(df), replace=True).reset_index(drop=True)
            X_boot = boot_df[features]
            y_boot = boot_df[target_column]

            # Train model on bootstrapped dataset
            # model = train_model(X_boot, y_boot, param_grid)
            model.fit(X_boot, y_boot)
            bootstrap_models.append(model)
            bootstrap_probs.append(model.predict_proba(X)[:, 1])
            probs = pd.DataFrame({f'{i}_bootstrap_probs':model.predict_proba(X)[:, 1]})
            predictions = pd.concat([predictions, probs], axis=1)
            bar()

    np.save(result_path + "/bootstrap_probs.npy",  np.array(bootstrap_probs))
    return bootstrap_models, np.array(bootstrap_probs), predictions

# Plot comparison of predicted probabilities
def plot_probability_comparison(original_probs, bootstrap_probs, lowess_2_5, lowess_97_5, n_bootstrap, figure_path, model_name=""):
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
    plt.xlabel("Estimated Risks of the Original Model")
    plt.ylabel("Estimated Risks of the Bootstrapped Models")
    plt.title(f"{model_name} — Predicted Probabilities: Original and {n_bootstrap} Bootstrapped Models" if model_name else f"Comparison of Predicted Probabilities: Original Model and {n_bootstrap} Bootstrapped Models")
    plt.grid(False)
    plt.legend()
    plt.savefig(figure_path + '/probability_comparison.png')
    plt.show()

def _calib_alpha_beta(y, probs, eps=1e-15):
    """
    Fit logit(y) = alpha + beta * logit(probs) via unpenalized binomial GLM.
    Returns (alpha, beta, fit_result).
    """
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)  # avoid 0/1 logits
    lp = np.log(p / (1.0 - p))                                 # linear predictor (logit)
    X = sm.add_constant(lp)                                    # [1, lp]
    equation = sm.GLM(y, X, family=sm.families.Binomial()).fit()
    alpha, beta = equation.params.tolist()
    return alpha, beta, equation

# Plot calibration curve for original and bootstrapped models
def plot_calibration_with_bootstrap(origin_predict, bootstrap_probs, y, n_bootstrap, figure_path, model_name="", n_bins=5):
    plt.figure(figsize=(10, 6))
    
    # Bootstrap models calibration curves
    for i, probs in enumerate(bootstrap_probs):
        mean_predicted_prob, observed_fraction = calibration_curve(y, probs, n_bins=n_bins, strategy='uniform')
        plt.plot(mean_predicted_prob, observed_fraction, color='grey', alpha=0.6, lw=1, label="Bootstrapped Models" if i == 0 else "")  # Label only the first curve
        # disp = CalibrationDisplay.from_predictions(y, bootstrap_probs)
    
    # Original model calibration curve
    mean_predicted_prob, observed_fraction = calibration_curve(y, origin_predict, n_bins=n_bins, strategy='uniform')
    plt.plot(mean_predicted_prob, observed_fraction, 'k--', lw=2, label="Original Model (Dashed Line)")

    # Ideal calibration line
    plt.plot([0, 1], [0, 1], 'k-', lw=1.5, label="Ideal Calibration Line")
    
    orig_CITL, orig_slope, _ = _calib_alpha_beta(y, origin_predict)
    # Plot settings
    plt.xlabel("Model's Predicted Probabilities")
    plt.ylabel("Observed Predicted Probabilities")
    plt.title(f"{model_name} — Calibration Instability with {n_bootstrap} Bootstrapped Models" if model_name else f"Model Calibration Instability with {n_bootstrap} Bootstrapped Models")
    plt.legend(loc='upper left', frameon=True, fontsize='small')
    plt.grid(True)
    plt.savefig(figure_path + '/calibration_comparison.png')
    plt.show()
    
    return orig_CITL, orig_slope

# Calculate LOWESS smoothed percentiles
def calculate_lowess_percentiles(bootstrap_probs, original_probs):
    percentile_2_5 = np.percentile(bootstrap_probs, 2.5, axis=0)
    percentile_97_5 = np.percentile(bootstrap_probs, 97.5, axis=0)
    lowess_2_5 = sm.nonparametric.lowess(percentile_2_5, original_probs, frac=0.3)
    lowess_97_5 = sm.nonparametric.lowess(percentile_97_5, original_probs, frac=0.3)
    return lowess_2_5, lowess_97_5

# Plot MAPE instability 
def plot_mape_instability(origin_predict, bootstrap_probs, figure_path, model_name=""):
    
    pred_probs_T = bootstrap_probs.T
    absolute_errors = np.abs(pred_probs_T - origin_predict[:, np.newaxis])
    # Calculate Mean Absolute Prediction Error (MAPE)
    mape = np.mean(absolute_errors, axis = 1)

    y_values = mape.flatten()
    # Repeat origin_predict values for each column in bootstrap_probs
    # x_values = np.repeat(origin_predict, absolute_errors.shape[1])
    x_values = origin_predict
    # Plot MAPE instability
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, alpha=0.5, s=0.3)
    # plt.axhline(mean_mape, color='red', linestyle='--', label=f"Mean MAPE: {mean_mape:.2f}%")
    plt.xlabel("Estimated Risk of the Original Model")
    plt.ylabel("MAPE")
    plt.ylim(0, 0.45)
    plt.xlim(0, 1)
    plt.title(f"{model_name} — MAPE Instability Plot" if model_name else "MAPE Instability Plot")
    plt.grid(True)
    plt.savefig(figure_path + '/mape.png')
    plt.show()
    return mape

    
def calculate_classification_instability(df, num_bootstraps, threshold=0.1):
    """
    Calculates classification instability based on bootstrap samples.

    Args:
        df (pd.DataFrame): DataFrame containing 'origin_predict' and '0_bootstrap_probs', '1_bootstrap_probs', ..., '199_bootstrap_probs'.
        threshold (float, optional): Risk threshold. Defaults to 0.1.
        num_bootstraps (int, optional): Number of bootstrap samples. Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame with 'pr_orig' and 'pr_change' (instability index).
    """

    df_copy = df.copy() #avoid modifying the original dataframe
    for i in range(num_bootstraps):
        change_col = f'pr{i}_change'
        df_copy[change_col] = 0
        df_copy.loc[((df_copy[f'{i}_bootstrap_probs'] >= threshold) & (df_copy['origin_predict'] < threshold)), change_col] = 1
        df_copy.loc[((df_copy[f'{i}_bootstrap_probs'] < threshold) & (df_copy['origin_predict'] >= threshold)), change_col] = 1

    change_cols = [f'pr{i}_change' for i in range(num_bootstraps)]
    
    # Per-patient CII: mean across bootstraps (axis=1)
    df_copy['pr_change'] = df_copy[change_cols].mean(axis=1)
    
    # Per-bootstrap CII: mean across patients (axis=0)
    cii_per_bootstrap = df_copy[change_cols].mean(axis=0).values
    
    return df_copy[['origin_predict', 'pr_change']], cii_per_bootstrap

def plot_classification_instability(df, figure_path, threshold=0.1, model_name=""):
    """
    Plots the classification instability against original predictions.

    Args:
        df (pd.DataFrame): DataFrame with 'origin_predict' and 'pr_change'.
        threshold (float, optional): Risk threshold. Defaults to 0.1.
    """
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="whitegrid") # Set the plot style

    # Scatter plot with jitter
    sns.scatterplot(x='origin_predict', y='pr_change', data=df, s=20, color='grey', alpha=0.7)

    # Labels and title
    plt.xlabel("Estimated Risk from Developed Model")
    plt.ylabel("Classification Instability Index")
    plt.title(f"{model_name} — Classification Instability and Original Predictions" if model_name else "Classification Instability and Original Predictions")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal') # set aspect ratio to 1 for a more square-like plot
    
    #set x and y axis ticks
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(0, 1.2, 0.2))
    # set figure background color to white
    plt.gca().set_facecolor('white')
    plt.savefig(figure_path + '/classification.png')
    # Show the plot
    plt.show()
    
    return df

# --- Cross-sample-size comparison plots (generic) ---

def plot_generic_boxplots(summary_data, key1, key2, label1, label2, color1, color2, title, figure_path, filename):
    """
    Generic side-by-side box plots comparing two metrics across sample sizes.
    Left y-axis = metric 1, Right y-axis = metric 2.
    """
    sorted_data = sorted(summary_data, key=lambda d: d['n'])
    labels = [str(d['n']) for d in sorted_data]
    dist1 = [d[key1] for d in sorted_data]
    dist2 = [d[key2] for d in sorted_data]
    n_groups = len(labels)

    fig, ax1 = plt.subplots(figsize=(max(10, n_groups * 2.5), 6))
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')

    group_centers = np.arange(n_groups) * 2.0
    offset = 0.4

    bp1 = ax1.boxplot(dist1, positions=group_centers - offset, widths=0.6,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='o', markersize=1, alpha=0.4))
    for patch in bp1['boxes']:
        patch.set_facecolor(color1)
        patch.set_alpha(0.7)
    for median in bp1['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    ax1.set_ylabel(label1, fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    ax2 = ax1.twinx()
    bp2 = ax2.boxplot(dist2, positions=group_centers + offset, widths=0.6,
                      patch_artist=True, showfliers=True,
                      flierprops=dict(marker='o', markersize=1, alpha=0.4))
    for patch in bp2['boxes']:
        patch.set_facecolor(color2)
        patch.set_alpha(0.7)
    for median in bp2['medians']:
        median.set_color('black')
        median.set_linewidth(1.5)
    ax2.set_ylabel(label2, fontsize=12, color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    ax1.set_xticks(group_centers)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel('Sample Size (n)', fontsize=12, color='black')
    ax1.set_title(title, fontsize=14, color='black')
    ax1.tick_params(axis='x', labelcolor='black')

    # Alternating background bands to group each pair
    band_width = 1.6  # slightly wider than the two boxes combined
    for i, center in enumerate(group_centers):
        if i % 2 == 0:
            ax1.axvspan(center - band_width/2, center + band_width/2,
                        color='#f0f0f0', zorder=0)

    from matplotlib.patches import Patch
    legend_items = [Patch(facecolor=color1, alpha=0.7, label=label1),
                    Patch(facecolor=color2, alpha=0.7, label=label2)]
    ax1.legend(handles=legend_items, loc='upper right', frameon=True).set_zorder(10)
    ax1.grid(False)
    ax2.grid(False)

    plt.tight_layout()
    plt.savefig(f'{figure_path}/{filename}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_generic_correlation_trend(summary_data, corr_key, p_key, ylabel, title, figure_path, filename, color='purple'):
    """
    Generic correlation trend plot across sample sizes.
    """
    df_plot = pd.DataFrame(summary_data).sort_values('n')

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot['n'], df_plot[corr_key], 'o-', color=color,
             linewidth=2, markersize=8)

    for _, row in df_plot.iterrows():
        plt.annotate(f"r = {row[corr_key]:.3f}",
                     (row['n'], row[corr_key]),
                     textcoords='offset points', xytext=(0, 12),
                     ha='center', fontsize=9)

    plt.xlabel('Sample Size (n)')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figure_path}/{filename}.png', dpi=150)
    plt.show()


def plot_optimism_vs_pct_mape(summary_data, figure_path, model_name=""):
    """
    Scatter + line plot: Mean Optimism (x) vs % MAPE < 5% (y),
    one point per sample size, labeled by n.
    """
    df_plot = pd.DataFrame(summary_data).sort_values('n')

    plt.figure(figsize=(10, 6))
    plt.plot(df_plot['mean_opt'], df_plot['pct_mape_lt_5'],
             'o-', color='#ff7f0e', linewidth=2, markersize=10)

    # Annotate each point with sample size
    for _, row in df_plot.iterrows():
        plt.annotate(f"n={row['n']:.0f}",
                     (row['mean_opt'], row['pct_mape_lt_5']),
                     textcoords='offset points', xytext=(8, 8),
                     ha='left', fontsize=9)

    plt.xlabel('Mean Optimism')
    plt.ylabel('% of Patients with MAPE < 5%')
    plt.title(f"{model_name} — Optimism and % MAPE < 5%" if model_name else "Optimism and % MAPE < 5%")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{figure_path}/optimism_vs_pct_mape_lt5.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_combined_probability_comparison(combined_data, figure_path, model_name=""):
    """
    Combined multi-panel figure of predicted probability comparison plots,
    one subplot per sample size. Saved as a single image file.

    Args:
        combined_data: list of dicts, each with keys:
            'n', 'original_probs', 'bootstrap_probs', 'lowess_2_5', 'lowess_97_5', 'n_bootstrap'
        figure_path: directory to save figure
        model_name: model label for suptitle
    """
    n_panels = len(combined_data)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()  # ensure iterable even for 1 panel

    for idx, entry in enumerate(sorted(combined_data, key=lambda d: -d['n'])):
        ax = axes[idx]
        for probs in entry['bootstrap_probs']:
            ax.scatter(entry['original_probs'], probs,
                       color='#ff7f0e', alpha=1, s=0.05)
        ax.plot([0, 1], [0, 1], 'k-', lw=1, label="Ideal Line")
        ax.plot(entry['lowess_2_5'][:, 0], entry['lowess_2_5'][:, 1],
                'k--', lw=1, label="LOWESS 2.5th")
        ax.plot(entry['lowess_97_5'][:, 0], entry['lowess_97_5'][:, 1],
                'k--', lw=1, label="LOWESS 97.5th")
        ax.set_xlabel("Estimated Risks (Original)", fontsize=9)
        ax.set_ylabel("Estimated Risks (Bootstrap)", fontsize=9)
        ax.set_title(f"n = {entry['n']}", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(False)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')

    # Hide unused axes
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{model_name} — Predicted Probabilities: Original and Bootstrapped Models"
        if model_name else "Predicted Probabilities: Original and Bootstrapped Models",
        fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{figure_path}/combined_probability_comparison.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()


def plot_combined_classification_instability(combined_data, figure_path, threshold=0.1, model_name=""):
    """
    Combined multi-panel figure of classification instability plots,
    one subplot per sample size. Saved as a single image file.

    Args:
        combined_data: list of dicts, each with keys:
            'n', 'instability_df' (DataFrame with 'origin_predict' and 'pr_change')
        figure_path: directory to save figure
        model_name: model label for suptitle
    """
    n_panels = len(combined_data)
    ncols = min(3, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5.5 * nrows))
    axes = np.array(axes).flatten()

    for idx, entry in enumerate(sorted(combined_data, key=lambda d: -d['n'])):
        ax = axes[idx]
        df_inst = entry['instability_df']
        ax.scatter(df_inst['origin_predict'], df_inst['pr_change'],
                   s=10, color='#ff7f0e', alpha=0.7)
        ax.set_xlabel("Estimated Risk (Original)", fontsize=9)
        ax.set_ylabel("Classification Instability Index", fontsize=9)
        ax.set_title(f"n = {entry['n']}", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.set_xticks(np.arange(0, 1.2, 0.2))
        ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.set_facecolor('white')

    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        f"{model_name} — Classification Instability and Original Predictions"
        if model_name else "Classification Instability and Original Predictions",
        fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{figure_path}/combined_classification_instability.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()


###########################################################

# Step 1: Define bootstraps and training configuration
param_grid = {
    'hidden_layer_sizes': [(8,), (16,), (32,), (64,), (32, 16), (64, 32), (64, 32, 16)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['adam'],  # 'lbfgs', 'sgd',
    'learning_rate': ['constant', 'adaptive'],
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'early_stopping': [True]
}

MODEL_NAME = "ANN"
n_bootstrap = 200
RANDOM_STATE = 931
df_path = r'dataset/gusto_dataset(Sheet1).csv'
df = pd.read_csv(df_path)
df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['pmi'] = df['pmi'].apply(lambda x: 1 if x == 'yes' else 0)
features = ['age', 'sex', 'hyp', 'htn', 'hrt', 'ste', 'pmi', 'sysbp']
key = 'day30'

###########################################################
# Step 3: SAMPLED DATASET
## Directory and data pre-processing
# random_state = 2569
# frac = [1]
#[1, 0.50, 0.25, 0.125, 0.05, 0.025, 0.0125, 0.01]
# frac = [1, 0.50, 0.25, 0.125, 0.05, 0.0125]

frac = [1, 0.50, 0.25, 0.125, 0.05, 0.0125]
# frac = [0.125, 0.05, 0.0125]
data_path = f'dataset/{MODEL_NAME}'
os.makedirs(data_path, exist_ok=True)

print(f"\n{'='*60}")
print(f"  Model: {MODEL_NAME}")
print(f"{'='*60}")

# Collect summary across sample sizes for cross-sample plots (1.1 & 1.2)
cross_sample_summary = []
combined_plot_data = []  # collect per-sample-size data for combined figures

for i in frac:
    if i == 1:
        df_sam = df
    else:
        df_sam = pd.concat([
            group.sample(frac=i, replace=False, random_state=RANDOM_STATE)
            for _, group in df.groupby(key)
        ]).reset_index(drop=True)
    print(f"\n--- {MODEL_NAME} | n={len(df_sam)} ---")
    
    df_sam.to_csv(data_path + f'/{len(df_sam)}_gustoShrinkedDf.csv')
    
    # ==========================Result Path==========================
    model_path = f'results/{MODEL_NAME}'
    result_path = f'{model_path}/df{len(df_sam)}'
    figure_path = f'{result_path}/figure'
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(figure_path, exist_ok=True)
    
    X = df_sam[features]
    y = df_sam[key]

    # ============================================================
    # INTERNAL VALIDATION (Bootstrap for C-statistic Optimism)
    # - Resamples data, trains model, evaluates on SAME bootstrap sample
    # - Measures discrimination optimism (AuROC)
    # ============================================================

    # Check if we can load cached results
    cached_probs_path = os.path.join(result_path, 'bootstrap_probs.npy')
    cached_predictions_path = os.path.join(result_path, 'full_predictions.csv')
    cached_optimism_path = os.path.join(result_path, 'optimism_values.npy')
    cached_output_path = os.path.join(result_path, 'output.json')

    if os.path.exists(cached_probs_path) and os.path.exists(cached_predictions_path) and os.path.exists(cached_optimism_path) and os.path.exists(cached_output_path):
        print(f"  ✓ Loading cached bootstrap results from {result_path}...")
        bootstrap_probs = np.load(cached_probs_path)
        predictions = pd.read_csv(cached_predictions_path, index_col=0)
        optimism = np.load(cached_optimism_path)
        
        # Load apparent_auc from output.json
        with open(cached_output_path, 'r') as f_json:
            out_data = json.load(f_json)
            apparent_auc = out_data.get('apparent_auc', 0.8) # fallback if not found
            
        origin_predict = predictions['origin_predict'].values
        c_bs = apparent_auc - optimism
        
        mean_c_bs = statistics.mean(c_bs)
        sd_c_bs = statistics.stdev(c_bs)
        ci_c_bs = [np.percentile(c_bs, 2.5), np.percentile(c_bs, 97.5)]
        
        mean_opt = statistics.mean(optimism)
        sd_c_opt = statistics.stdev(optimism)
        ci_c_opt = [np.percentile(optimism, 2.5), np.percentile(optimism, 97.5)]
        median_opt = np.median(optimism)
        iqr_opt = np.percentile(optimism, 75) - np.percentile(optimism, 25)
        p5_opt = np.percentile(optimism, 5)
        p95_opt = np.percentile(optimism, 95)
        
        plot_histogram_with_density(c_bs, figure_path, model_name=MODEL_NAME, title="C-statistic optimism (Original)")
        bootstrap_models = []
    else:
        ## Train original model
        model = train_model(X, y, param_grid)
        model.fit(X, y)
        origin_predict = model.predict_proba(X)[:, 1]
        apparent_auc = roc_auc_score(y, origin_predict)
        # np.save(result_path + "/origin_predict.npy",  np.array(origin_predict))

        c_bs = c_statistic_optimism(model, n_bootstrap, X, y)
        mean_c_bs = statistics.mean(c_bs)
        sd_c_bs = statistics.stdev(c_bs)
        ci_c_bs = [np.percentile(c_bs, 2.5), np.percentile(c_bs, 97.5)]

        print(f"[{MODEL_NAME}] C-statistic optimism - Mean: {mean_c_bs:.4f}, SD: {sd_c_bs:.4f}, 95%CI: {ci_c_bs[0]:.4f} - {ci_c_bs[1]:.4f}")

        with open(f"{result_path}/output.txt", "w") as file:
            file.write(f"[{MODEL_NAME}] C-statistic optimism - Mean: {mean_c_bs:.4f}, SD: {sd_c_bs:.4f}, 95%CI: {ci_c_bs[0]:.4f} - {ci_c_bs[1]:.4f}")

        optimism = []
        for i in c_bs:
            opt_boot = i - apparent_auc
            optimism.append(opt_boot)
        
        optimism = np.absolute(optimism)
        np.save(result_path + '/optimism_values.npy', np.array(optimism))

        mean_opt = statistics.mean(optimism)
        sd_c_opt = statistics.stdev(optimism)
        ci_c_opt = [np.percentile(optimism, 2.5), np.percentile(optimism, 97.5)]
        median_opt = np.median(optimism)
        iqr_opt = np.percentile(optimism, 75) - np.percentile(optimism, 25)
        p5_opt = np.percentile(optimism, 5)
        p95_opt = np.percentile(optimism, 95)
            
        plot_histogram_with_density(c_bs, figure_path, model_name=MODEL_NAME, title="C-statistic optimism (Original)")

        # ============================================================
        # MODEL INSTABILITY (Bootstrap for Prediction Stability)
        # - Resamples data, trains NEW models, predicts on ORIGINAL data
        # - Measures how predictions change with different training data
        # ============================================================

        ## Bootstrap training
        bootstrap_models, bootstrap_probs, predictions = bootstrap_training(model, X, df_sam, features, key, param_grid, n_bootstrap, origin_predict, result_path)

        predictions.to_csv(result_path + '/full_predictions.csv')

    ## Calculate LOWESS smoothed percentiles
    lowess_2_5, lowess_97_5 = calculate_lowess_percentiles(bootstrap_probs, origin_predict)

    ## Plot results
    mape = plot_mape_instability(origin_predict, bootstrap_probs, figure_path, model_name=MODEL_NAME)

    mean_mape = statistics.mean(mape)
    sd_c_mape = statistics.stdev(mape)
    ci_c_mape = [np.percentile(mape, 2.5), np.percentile(mape, 97.5)]
    median_mape = np.median(mape)
    iqr_mape = np.percentile(mape, 75) - np.percentile(mape, 25)
    p5_mape = np.percentile(mape, 5)
    p95_mape = np.percentile(mape, 95)

    # Percentage of patients with MAPE < 5%
    pct_mape_lt_5 = np.mean(mape < 0.05) * 100
        
    plot_probability_comparison(origin_predict, bootstrap_probs, lowess_2_5, lowess_97_5, n_bootstrap, figure_path, model_name=MODEL_NAME)
    orig_CITL, orig_slope = plot_calibration_with_bootstrap(origin_predict, bootstrap_probs, y, n_bootstrap, figure_path, model_name=MODEL_NAME)
    instability_df, cii_per_bootstrap = calculate_classification_instability(predictions, n_bootstrap, 0.1)
    # instability_df.to_csv(result_path + '/cii_df.csv')
    np.save(result_path + '/cii_values.npy', np.array(cii_per_bootstrap))
    plot_classification_instability(instability_df, figure_path, 0.1, model_name=MODEL_NAME)

    # Store data for combined multi-panel figures
    combined_plot_data.append({
        'n': len(df_sam),
        'original_probs': origin_predict,
        'bootstrap_probs': bootstrap_probs,
        'lowess_2_5': lowess_2_5,
        'lowess_97_5': lowess_97_5,
        'n_bootstrap': n_bootstrap,
        'instability_df': instability_df,
    })

    cii = instability_df['pr_change']

    mean_cii = statistics.mean(cii)
    sd_cii = statistics.stdev(cii)
    ci_cii = [np.percentile(cii, 2.5), np.percentile(cii, 97.5)]
    median_cii = np.median(cii)
    iqr_cii = np.percentile(cii, 75) - np.percentile(cii, 25)
    p5_cii = np.percentile(cii, 5)
    p95_cii = np.percentile(cii, 95)

    # --- Per-bootstrap MAPE (for correlation with optimism) ---
    # Each bootstrap i gives: mean |bootstrap_probs[i] - origin_predict| across all patients
    mape_per_bootstrap = np.mean(np.abs(bootstrap_probs - origin_predict[np.newaxis, :]), axis=1)
    corr_opt_mape, p_opt_mape = pearsonr(optimism, mape_per_bootstrap)
    print(f"  Correlation(Optimism, MAPE): r={corr_opt_mape:.4f}, p={p_opt_mape:.4e}")

    # --- Per-bootstrap CII correlation with optimism (reusing cii_per_bootstrap from above) ---
    corr_opt_cii, p_opt_cii = pearsonr(optimism, cii_per_bootstrap)
    print(f"  Correlation(Optimism, CII):  r={corr_opt_cii:.4f}, p={p_opt_cii:.4e}")

    # --- Per-bootstrap CITL and Calibration Slope ---
    citl_per_bootstrap = []
    slope_per_bootstrap = []
    for b in range(n_bootstrap):
        try:
            alpha_b, beta_b, _ = _calib_alpha_beta(y, bootstrap_probs[b])
            citl_per_bootstrap.append(alpha_b)
            slope_per_bootstrap.append(beta_b)
        except Exception:
            citl_per_bootstrap.append(np.nan)
            slope_per_bootstrap.append(np.nan)
    citl_per_bootstrap = np.array(citl_per_bootstrap)
    slope_per_bootstrap = np.array(slope_per_bootstrap)

    # Remove NaNs for correlation computation
    valid_mask = ~(np.isnan(citl_per_bootstrap) | np.isnan(slope_per_bootstrap))
    citl_valid = citl_per_bootstrap[valid_mask]
    slope_valid = slope_per_bootstrap[valid_mask]
    mape_valid = mape_per_bootstrap[valid_mask]
    cii_valid = cii_per_bootstrap[valid_mask]

    # --- Correlations: MAPE/CII vs CITL/Slope ---
    corr_mape_citl, p_corr_mape_citl = pearsonr(mape_valid, citl_valid)
    corr_mape_slope, p_corr_mape_slope = pearsonr(mape_valid, slope_valid)
    corr_cii_citl, p_corr_cii_citl = pearsonr(cii_valid, citl_valid)
    corr_cii_slope, p_corr_cii_slope = pearsonr(cii_valid, slope_valid)
    print(f"  Correlation(MAPE, CITL):  r={corr_mape_citl:.4f}, p={p_corr_mape_citl:.4e}")
    print(f"  Correlation(MAPE, Slope): r={corr_mape_slope:.4f}, p={p_corr_mape_slope:.4e}")
    print(f"  Correlation(CII, CITL):   r={corr_cii_citl:.4f}, p={p_corr_cii_citl:.4e}")
    print(f"  Correlation(CII, Slope):  r={corr_cii_slope:.4f}, p={p_corr_cii_slope:.4e}")

    # Collect for cross-sample-size plots (includes raw distributions for box plots)
    cross_sample_summary.append({
        'n': len(df_sam),
        'mean_opt': mean_opt,
        'sd_opt': sd_c_opt,
        'median_opt': median_opt,
        'iqr_opt': iqr_opt,
        'p5_opt': p5_opt,
        'p95_opt': p95_opt,
        'mean_mape': mean_mape,
        'sd_mape': sd_c_mape,
        'median_mape': median_mape,
        'iqr_mape': iqr_mape,
        'p5_mape': p5_mape,
        'p95_mape': p95_mape,
        'mean_cii': mean_cii,
        'sd_cii': sd_cii,
        'median_cii': median_cii,
        'iqr_cii': iqr_cii,
        'p5_cii': p5_cii,
        'p95_cii': p95_cii,
        'correlation': corr_opt_mape,
        'p_opt_mape': p_opt_mape,
        'corr_opt_cii': corr_opt_cii,
        'p_opt_cii': p_opt_cii,
        'corr_mape_citl': corr_mape_citl, 'p_mape_citl': p_corr_mape_citl,
        'corr_mape_slope': corr_mape_slope, 'p_mape_slope': p_corr_mape_slope,
        'corr_cii_citl': corr_cii_citl, 'p_cii_citl': p_corr_cii_citl,
        'corr_cii_slope': corr_cii_slope, 'p_cii_slope': p_corr_cii_slope,
        'pct_mape_lt_5': pct_mape_lt_5,
        'optimism_values': list(optimism),
        'mape_values': list(mape),
        'cii_values': list(cii_per_bootstrap),
        'citl_values': list(citl_per_bootstrap),
        'slope_values': list(slope_per_bootstrap),
    })

    output_data = {
        "model": MODEL_NAME,
        "dataset": len(df_sam),
        "apparent_auc": apparent_auc,
        "mean_auc": round(mean_c_bs, 4),
        "sd_auc": round(sd_c_bs, 4),
        "ci_95_auc": [round(ci_c_bs[0], 4), round(ci_c_bs[1], 4)],
        "mean_optimism": round(mean_opt, 4),
        "sd_optimism": round(sd_c_opt, 4),
        "ci_95_optimism": [round(ci_c_opt[0], 4), round(ci_c_opt[1], 4)],
        "median_optimism": round(median_opt, 4),
        "iqr_optimism": round(iqr_opt, 4),
        "p5_optimism": round(p5_opt, 4),
        "p95_optimism": round(p95_opt, 4),
        "calibrationSlope": orig_slope,
        "CITL" : orig_CITL,
        "mean_mape": round(mean_mape, 4),  
        "sd_c_mape": round(sd_c_mape, 4),
        "ci_c_mape": [round(ci_c_mape[0], 4), round(ci_c_mape[1], 4)],
        "median_mape": round(median_mape, 4),
        "iqr_mape": round(iqr_mape, 4),
        "p5_mape": round(p5_mape, 4),
        "p95_mape": round(p95_mape, 4),
        "mean_cii": round(mean_cii, 4),
        "sd_cii": round(sd_cii, 4),
        "ci_cii": [round(ci_cii[0], 4), round(ci_cii[1], 4)],
        "median_cii": round(median_cii, 4),
        "iqr_cii": round(iqr_cii, 4),
        "p5_cii": round(p5_cii, 4),
        "p95_cii": round(p95_cii, 4),
        "corr_optimism_mape": round(corr_opt_mape, 4),
        }

    with open(f"{result_path}/output.json", "w") as file:
        json.dump(output_data, file, indent=4)

# ==============================================================
# CROSS-SAMPLE-SIZE COMPARISON PLOTS
# ==============================================================
os.makedirs(f'{model_path}/cross_sample_figures', exist_ok=True)
fig_dir = f'{model_path}/cross_sample_figures'

# --- Combined multi-panel figures ---
plot_combined_probability_comparison(combined_plot_data, fig_dir, model_name=MODEL_NAME)
plot_combined_classification_instability(combined_plot_data, fig_dir, model_name=MODEL_NAME)

# --- Optimism vs MAPE ---
plot_generic_boxplots(cross_sample_summary, 'optimism_values', 'mape_values',
    'Optimism', 'MAPE', '#ff7f0e', '#ffeacf',
    f'{MODEL_NAME} — Optimism and MAPE across Sample Sizes', fig_dir, 'optimism_vs_mape')
plot_generic_correlation_trend(cross_sample_summary, 'correlation', 'p_opt_mape',
    'Pearson r (Optimism and MAPE)', f'{MODEL_NAME} — Correlation: Optimism and MAPE across Sample Sizes',
    fig_dir, 'optimism_mape_correlation', color='#ff7f0e')

# --- Optimism vs % MAPE < 5% ---
plot_optimism_vs_pct_mape(cross_sample_summary, fig_dir, model_name=MODEL_NAME)

# --- Optimism vs CII ---
plot_generic_boxplots(cross_sample_summary, 'optimism_values', 'cii_values',
    'Optimism', 'CII', '#ff7f0e', '#ffeacf',
    f'{MODEL_NAME} — Optimism and CII across Sample Sizes', fig_dir, 'optimism_vs_cii')
plot_generic_correlation_trend(cross_sample_summary, 'corr_opt_cii', 'p_opt_cii',
    'Pearson r (Optimism and CII)', f'{MODEL_NAME} — Correlation: Optimism and CII across Sample Sizes',
    fig_dir, 'optimism_cii_correlation', color='#ff7f0e')

# # --- Optimism vs MAPE 5% ---
# plot_generic_boxplots(cross_sample_summary, 'optimism_values', 'pct_mape_lt_5',
#     'Optimism', '% MAPE < 5%', 'steelblue', 'coral',
#     f'{MODEL_NAME} — Optimism vs % MAPE < 5% across Sample Sizes', fig_dir, 'optimism_vs_pct_mape_lt_5')
# plot_generic_correlation_trend(cross_sample_summary, 'corr_opt_pct_mape', 'p_opt_pct_mape',
#     'Pearson r (Optimism vs % MAPE < 5%)', f'{MODEL_NAME} — Correlation: Optimism vs % MAPE < 5% across Sample Sizes',
#     fig_dir, 'optimism_pct_mape_lt_5_correlation', color='purple')

# # --- CITL vs MAPE ---
# plot_generic_boxplots(cross_sample_summary, 'citl_values', 'mape_values',
#     'CITL', 'MAPE', 'goldenrod', 'coral',
#     f'{MODEL_NAME} — CITL vs MAPE across Sample Sizes', fig_dir, 'citl_vs_mape')
# plot_generic_correlation_trend(cross_sample_summary, 'corr_mape_citl', 'p_mape_citl',
#     'Pearson r (CITL vs MAPE)', f'{MODEL_NAME} — Correlation: CITL vs MAPE across Sample Sizes',
#     fig_dir, 'citl_mape_correlation', color='goldenrod')

# # --- CITL vs CII ---
# plot_generic_boxplots(cross_sample_summary, 'citl_values', 'cii_values',
#     'CITL', 'CII', 'goldenrod', 'seagreen',
#     f'{MODEL_NAME} — CITL vs CII across Sample Sizes', fig_dir, 'citl_vs_cii')
# plot_generic_correlation_trend(cross_sample_summary, 'corr_cii_citl', 'p_cii_citl',
#     'Pearson r (CITL vs CII)', f'{MODEL_NAME} — Correlation: CITL vs CII across Sample Sizes',
#     fig_dir, 'citl_cii_correlation', color='darkgoldenrod')

# # --- Slope vs MAPE ---
# plot_generic_boxplots(cross_sample_summary, 'slope_values', 'mape_values',
#     'Calibration Slope', 'MAPE', 'mediumpurple', 'coral',
#     f'{MODEL_NAME} — Calibration Slope vs MAPE across Sample Sizes', fig_dir, 'slope_vs_mape')
# plot_generic_correlation_trend(cross_sample_summary, 'corr_mape_slope', 'p_mape_slope',
#     'Pearson r (Slope vs MAPE)', f'{MODEL_NAME} — Correlation: Calibration Slope vs MAPE across Sample Sizes',
#     fig_dir, 'slope_mape_correlation', color='mediumpurple')

# # --- Slope vs CII ---
# plot_generic_boxplots(cross_sample_summary, 'slope_values', 'cii_values',
#     'Calibration Slope', 'CII', 'mediumpurple', 'seagreen',
#     f'{MODEL_NAME} — Calibration Slope vs CII across Sample Sizes', fig_dir, 'slope_vs_cii')
# plot_generic_correlation_trend(cross_sample_summary, 'corr_cii_slope', 'p_cii_slope',
#     'Pearson r (Slope vs CII)', f'{MODEL_NAME} — Correlation: Calibration Slope vs CII across Sample Sizes',
#     fig_dir, 'slope_cii_correlation', color='indigo')

# --- Export cross-sample summary (with p-values) to CSV ---
export_cols = [
    'n', 
    'mean_opt', 'sd_opt', 'median_opt', 'iqr_opt', 'p5_opt', 'p95_opt',
    'mean_mape', 'sd_mape', 'median_mape', 'iqr_mape', 'p5_mape', 'p95_mape',
    'mean_cii', 'sd_cii', 'median_cii', 'iqr_cii', 'p5_cii', 'p95_cii',
    'pct_mape_lt_5',
    'correlation', 'p_opt_mape', 'corr_opt_cii', 'p_opt_cii',
    'corr_mape_citl', 'p_mape_citl', 'corr_mape_slope', 'p_mape_slope',
    'corr_cii_citl', 'p_cii_citl', 'corr_cii_slope', 'p_cii_slope'
]

df_export = pd.DataFrame(cross_sample_summary)[export_cols].sort_values('n')
df_export.to_csv(f'{fig_dir}/cross_sample_summary.csv', index=False)
df_export.to_csv(f'{result_path}/cross_sample_summary.csv', index=False)
print(f"\nCross-sample summary exported to {fig_dir}/cross_sample_summary.csv and {result_path}/cross_sample_summary.csv")

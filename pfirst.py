"""
Created Date:       15/1/2024
Last update:        23/1/2024 (Refined by Gemini)
Program:            Class imbalance correction method and prediction stability
@author:            Asst. Prof. Wachiranun Sirikul, MD., M.Sc.
Description:        Class imbalance correction method and prediction stability.
                    This version refactors the code for improved readability, 
                    efficiency, and adds detailed documentation.
"""

"""
Part 1: Package installation (if needed)
Part 2: Library and function import
Part 3: Data import from excel file
Part 4: Rebalancing data
Part 5: Predictor selection (not used in this version)
Part 6: Original model development
Part 7: C-statistic optimism
Part 8: Model stability
"""

"""Part 2: Library and function import"""
# General packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
import statistics
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
import os

# Modeling packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Data imputation (not used in this version)
from sklearn.impute import KNNImputer

# Feature selection (not used in this version)
from sklearn.inspection import permutation_importance


# Rebalancing methods
from imblearn.over_sampling import SMOTENC, ADASYN, BorderlineSMOTE

# Progress bar
from alive_progress import alive_bar

# --- Set random state for reproducibility
RANDOM_STATE = 1


# --- Helper Functions ---
def knn_impute(data_x, n_neighbors=5):
    """
    Imputes missing values using KNNImputer.

    Args:
        data_x (pd.DataFrame): Input data with missing values.
        n_neighbors (int): Number of neighbors for imputation.

    Returns:
        np.ndarray: Imputed data.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors)
    return imputer.fit_transform(data_x)


def permutation_feature_selection(x, y, n_splits=10, n_repeats=15):
    """
    Performs permutation-based feature importance using Stratified K-Fold cross-validation.

    Args:
        x (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        n_splits (int): Number of folds for cross-validation.
        n_repeats (int): Number of repeats for permutation importance.

    Returns:
        pd.DataFrame: DataFrame with mean and std of feature importances.
    """
    model = LogisticRegression(random_state=RANDOM_STATE)
    skf = StratifiedKFold(n_splits=n_splits, random_state=RANDOM_STATE, shuffle=True)
    feature_importances = []

    for train_index, test_index in skf.split(x, y):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        model.fit(x_train, y_train)
        
        features_important = permutation_importance(
            model, x_test, y_test, n_repeats=n_repeats, random_state=RANDOM_STATE
        )
        fold_features_df = pd.DataFrame({
            "importances_mean": features_important["importances_mean"],
            "importances_std": features_important["importances_std"]
        }, index=x.columns)
        feature_importances.append(fold_features_df)
    
    combined_importances = pd.concat(feature_importances).groupby(level=0).mean()
    combined_importances = combined_importances.sort_values(by="importances_mean", ascending=False)
    return combined_importances


def grid_search_logistic_regression(x, y, penalty='l2'):
    """
    Performs GridSearchCV for logistic regression with specified penalty.

    Args:
        x (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        penalty (str): Type of penalty to use ('l1', 'l2', 'elasticnet', 'none').

    Returns:
        LogisticRegression: Fitted logistic regression model.
    """
    model = LogisticRegression(random_state=RANDOM_STATE)
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'max_iter': [1000],
        'solver': ["newton-cholesky", "sag", "saga", "lbfgs"],
        'penalty': [penalty] if penalty else ["none"]
    }
    if penalty == 'elasticnet':
        param_grid['l1_ratio'] = [0, 0.25, 0.5, 0.75, 1]
    
    gcv = GridSearchCV(model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=0, return_train_score=True) #verbose was 2, changed to 0 to remove noise
    gcv.fit(x, y)
    model.set_params(**gcv.best_params_)
    return model


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


def prediction_stability(model, n_bootstrap, x, y, prob_original_df):
    """
    Evaluates prediction stability using bootstrap samples.

    Args:
        model (sklearn model): Trained model.
        n_bootstrap (int): Number of bootstrap samples.
        x (pd.DataFrame): Feature data.
        y (pd.Series): Target data.
        prob_original_df (pd.DataFrame): DataFrame with original predictions.

    Returns:
        tuple: List of c-statistics and DataFrame with bootstrap predictions.
    """
    data = pd.concat([x, y], axis=1)
    c_stat_stability = []
    prob_df = prob_original_df

    with alive_bar(n_bootstrap, title="Prediction Stability Bootstrap") as bar:
        for i in range(n_bootstrap):
            data_sample = data.sample(n=len(data), replace=True).reset_index(drop=True)
            x_sample, y_sample = data_sample.iloc[:, :-1], data_sample.iloc[:, -1]
            model.fit(x_sample, y_sample)
            c = roc_auc_score(y, model.predict_proba(x)[:, 1])
            c_stat_stability.append(c)
            prob_new = pd.Series(model.predict_proba(x)[:, 1], name=f'prob_{i}')
            prob_df = pd.concat([prob_df, prob_new], axis=1)
            bar()
    return c_stat_stability, prob_df


# --- Plotting Functions ---
def plot_histogram_with_density(data, title="C-statistics stability"):
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
    plt.title(title)
    plt.legend()
    plt.savefig(results + '/optimism.png')
    plt.show()


def plot_prediction_instability(prob_df, n_bootstrap, title="Prediction Instability"):
    """
    Plots prediction instability using scatter plots and percentile lines.

    Args:
        prob_df (pd.DataFrame): DataFrame containing original and bootstrap predictions.
        n_bootstrap (int): Number of bootstrap samples.
        title (str): Title of the plot.
    """
    
    df = prob_df
    bootstrap_columns = [f'prob_{i}' for i in range(n_bootstrap)]
    df['lower'] = df[bootstrap_columns].apply(lambda row: np.percentile(row, 2.5), axis=1)
    df['upper'] = df[bootstrap_columns].apply(lambda row: np.percentile(row, 97.5), axis=1)
    pr_original = df['prob_original']
    plt.figure(figsize=(10, 10))
    for col in bootstrap_columns:
        plt.scatter(pr_original, df[col], color='#608BC1', alpha=0.3, s=5)
    
    sns.regplot(x=pr_original, y=df['upper'], scatter=False, color='black', 
                line_kws={'linestyle': 'dashed', 'linewidth': 1.5}, lowess=True)
    sns.regplot(x=pr_original, y=df['lower'], scatter=False, color='black', 
                line_kws={'linestyle': 'dashed', 'linewidth': 1.5}, lowess=True)
    
    plt.plot([pr_original.min(), pr_original.max()], [pr_original.min(), pr_original.max()], 
             color='black', linestyle='solid')
    
    plt.xlabel("Estimated risk from developed model")
    plt.xticks(np.arange(0.0, 1.1, 0.1))
    plt.yticks(np.arange(0.0, 1.1, 0.1))
    plt.ylabel("Estimated risk from bootstrap models")
    plt.title(title)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    plt.savefig(results + '/prediction_stability.png')
    plt.show()


def plot_mape(prob_df, n_bootstrap, title="MAPE over Individuals"):
    """
    Plots Mean Absolute Prediction Error (MAPE) against original predictions.

    Args:
        prob_df (pd.DataFrame): DataFrame containing original and bootstrap predictions.
        n_bootstrap (int): Number of bootstrap samples.
        title (str): Title of the plot.
    """
    df = prob_df
    for i in range(n_bootstrap):
        df[f'error_bs{i}'] = df[f'prob_{i}'] - df['prob_original']
        df[f'abs_error_bs{i}'] = df[f'error_bs{i}'].abs()
    
    abs_error_columns = [f'abs_error_bs{i}' for i in range(n_bootstrap)]
    df['MAPE'] = df[abs_error_columns].mean(axis=1)
    
    mape_summary = df['MAPE'].describe()
    print(mape_summary)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(df['prob_original'], df['MAPE'], color='gray', alpha=0.5, s=1)
    
    plt.xlabel("Estimated risk from developed model")
    plt.ylabel("Mean Absolute Prediction Error (MAPE)")
    plt.title(title)
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 0.50, 0.05))
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(False)
    
    plt.savefig(results + '/mape.png')
    plt.show()


def plot_calibration_stability(prob_df, outcome_data, n_bootstrap, title="Calibration stability plot"):
    """
    Plots calibration curves for original and bootstrap predictions.

    Args:
        prob_df (pd.DataFrame): DataFrame containing original and bootstrap predictions.
        outcome_data (pd.Series): True outcome data.
        n_bootstrap (int): Number of bootstrap samples.
        title (str): Title of the plot.
    """
    prob_df.sort_index(inplace=True)
    df_calibration = prob_df
    df_calibration['outcome'] = np.array(outcome_data)
    
    plt.figure(figsize=(8, 6))
    for i in range(n_bootstrap):
        prob_true, prob_pred = calibration_curve(df_calibration['outcome'], df_calibration[f'prob_{i}'], n_bins=10)
        plt.plot(prob_pred, prob_true, color='#CBDCEB', linewidth=0.5)
    
    prob_true, prob_pred = calibration_curve(df_calibration['outcome'], df_calibration['prob_original'], n_bins=10)
    plt.plot(prob_pred, prob_true, color="#133E87", linewidth=1, label=" Original calibration curve")
    plt.plot([0, 1], [0, 1], linestyle="--", color="#D76C82", label="Perfectly calibrated")
    plt.xlabel("Predicted probability")
    plt.ylabel("Actual probability")
    plt.title(title)
    plt.savefig(results + '/calibration.png')
    plt.legend()
    plt.show()
    
def calculate_classification_instability(df, threshold=0.100, num_bootstraps=200):
    """
    Calculates classification instability based on bootstrap samples.

    Args:
        df (pd.DataFrame): DataFrame containing 'pr_orig' and 'pr1', 'pr2', ..., 'pr200'.
        threshold (float, optional): Risk threshold. Defaults to 0.1.
        num_bootstraps (int, optional): Number of bootstrap samples. Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame with 'pr_orig' and 'pr_change' (instability index).
    """

    df_copy = df.copy() #avoid modifying the original dataframe
    for i in range(num_bootstraps):
        change_col = f'pr{i}_change'
        df_copy[change_col] = 0
        df_copy.loc[((df_copy[f'prob_{i}'] >= threshold) & (df_copy['prob_original'] < threshold)), change_col] = 1
        df_copy.loc[((df_copy[f'prob_{i}'] < threshold) & (df_copy['prob_original'] >= threshold)), change_col] = 1

    change_cols = [f'pr{i}_change' for i in range(num_bootstraps)]
    
    #Calculate difference betwee first and last changes
    df_copy['pr_change'] = df_copy[change_cols].mean(axis=1)
    

    return df_copy[['prob_original', 'pr_change']]


def plot_classification_instability(df, threshold = 0.100):
    """
    Plots the classification instability against original predictions.

    Args:
        df (pd.DataFrame): DataFrame with 'prob_original' and 'pr_change'.
        threshold (float, optional): Risk threshold. Defaults to 0.1.
    """
    plt.figure(figsize=(8, 8))
    sns.set_theme(style="whitegrid") # Set the plot style

    # Scatter plot with jitter
    sns.scatterplot(x='prob_original', y='pr_change', data=df, s=20, color='grey', alpha=0.7)

    # Labels and title
    plt.xlabel("Estimated Risk from Developed Model")
    plt.ylabel("Classification Instability Index")
    plt.title("Classification Instability vs Original Predictions")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.gca().set_aspect('equal') # set aspect ratio to 1 for a more square-like plot
    
    #set x and y axis ticks
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(0, 1.2, 0.2))
    # set figure background color to white
    plt.gca().set_facecolor('white')
    plt.savefig(results + '/classification.png')
    # Show the plot
    plt.show()

    
# --- Main Analysis ---

"""Part 3: Data import"""
# Import data file - CHANGE PATH AS NEEDED
try:
    df = pd.read_csv("/home/natthanaphop.isa/model_instability/dataset/gusto_dataset(Sheet1).csv")
except FileNotFoundError:
    print("Error: The specified data file was not found. Please check the file path.")
    exit()

df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['pmi'] = df['pmi'].apply(lambda x: 1 if x == 'yes' else 0)

# Keep the outcome (day30) and the 7 predictors of interest
x = df[["age", "ste", "htn", "sex", "hyp", "hrt", "pmi","sysbp"]]
y = df["day30"]

results = '/home/natthanaphop.isa/model_instability/results/instability2/full'
os.makedirs(results, exist_ok=True)

# Check event numbers
print("Original outcome distribution:\n", y.value_counts())

# small sample
# data = pd.concat([x, y], axis=1) 
# data_sample = data.sample(n=1100, replace=False).reset_index(drop=True)
# xs, ys = data_sample.iloc[:, :-1], data_sample.iloc[:, -1]

# Check event numbers
print("Original outcome distribution:\n", y.value_counts())

"""Part 4: Rebalancing data"""
# Initialize rebalancing methods
# smnc = SMOTENC(random_state=RANDOM_STATE, categorical_features=[2, 3, 4, 5, 6])
# bor = BorderlineSMOTE(random_state=RANDOM_STATE)
# ada = ADASYN(random_state=RANDOM_STATE)

# Apply rebalancing to create new datasets
# x_smnc, y_smnc = smnc.fit_resample(x, y)
# x_bor, y_bor = bor.fit_resample(x, y)
# x_ada, y_ada = ada.fit_resample(x, y)
# print("SMOTENC outcome distribution:\n",y_smnc.value_counts())
# print("BorderlineSMOTE outcome distribution:\n",y_bor.value_counts())
# print("ADASYN outcome distribution:\n",y_ada.value_counts())

"""Part 6: Original model development"""
# --- Model Training ---
# Train logistic regression models using grid search
model = grid_search_logistic_regression(x, y)
# model_smnc = grid_search_logistic_regression(x_smnc, y_smnc)
# model_bor = grid_search_logistic_regression(x_bor, y_bor)
# model_ada = grid_search_logistic_regression(x_ada, y_ada)

# Fit the models using the best parameters from grid search
model.fit(x, y)
# model_smnc.fit(x_smnc, y_smnc)
# model_bor.fit(x_bor, y_bor)
# model_ada.fit(x_ada, y_ada)

"""Part 7: C-statistic optimism"""
# set Boostrapping numbers
BS_number = 200
# --- C-statistic Optimism Calculation ---
# Calculate and print C-statistic optimism
c_bs = c_statistic_optimism(model, BS_number, x, y)
mean_c_bs = statistics.mean(c_bs)
sd_c_bs = statistics.stdev(c_bs)
ci_c_bs = sms.DescrStatsW(c_bs).tconfint_mean()
print(f"Original model C-statistic optimism - Mean: {mean_c_bs:.4f}, SD: {sd_c_bs:.4f}, 95%CI: {ci_c_bs[0]:.4f} - {ci_c_bs[1]:.4f}")
plot_histogram_with_density(c_bs, "C-statistic optimism (Original)")

# c_bs_smnc = c_statistic_optimism(model_smnc, BS_number, x, y)
# mean_c_bs_smnc = statistics.mean(c_bs_smnc)
# sd_c_bs_smnc = statistics.stdev(c_bs_smnc)
# ci_c_bs_smnc = sms.DescrStatsW(c_bs_smnc).tconfint_mean()
# print(f"SMOTENC model C-statistic optimism - Mean: {mean_c_bs_smnc:.4f}, SD: {sd_c_bs_smnc:.4f}, 95%CI: {ci_c_bs_smnc[0]:.4f} - {ci_c_bs_smnc[1]:.4f}")
# plot_histogram_with_density(c_bs_smnc, "C-statistic optimism (SMOTENC)")

# c_bs_bor = c_statistic_optimism(model_bor, BS_number, x, y)
# mean_c_bs_bor = statistics.mean(c_bs_bor)
# sd_c_bs_bor = statistics.stdev(c_bs_bor)
# ci_c_bs_bor = sms.DescrStatsW(c_bs_bor).tconfint_mean()
# print(f"BorderlineSMOTE model C-statistic optimism - Mean: {mean_c_bs_bor:.4f}, SD: {sd_c_bs_bor:.4f}, 95%CI: {ci_c_bs_bor[0]:.4f} - {ci_c_bs_bor[1]:.4f}")
# plot_histogram_with_density(c_bs_bor, "C-statistic optimism (BorderlineSMOTE)")


# c_bs_ada = c_statistic_optimism(model_ada, BS_number, x, y)
# mean_c_bs_ada = statistics.mean(c_bs_ada)
# sd_c_bs_ada = statistics.stdev(c_bs_ada)
# ci_c_bs_ada = sms.DescrStatsW(c_bs_ada).tconfint_mean()
# print(f"ADASYN model C-statistic optimism - Mean: {mean_c_bs_ada:.4f}, SD: {sd_c_bs_ada:.4f}, 95%CI: {ci_c_bs_ada[0]:.4f} - {ci_c_bs_ada[1]:.4f}")
# plot_histogram_with_density(c_bs_ada, "C-statistic optimism (ADASYN)")

"""Part 8: Model stability"""
# --- Model Stability Analysis ---
# Original Model
prob_original = pd.DataFrame(model.predict_proba(x)[:, 1], columns=['prob_original'])
c_sta, pred_sta = prediction_stability(model, BS_number, x, y, prob_original)
plot_prediction_instability(pred_sta, BS_number, "Prediction Instability (Original)")
plot_mape(pred_sta, BS_number, "MAPE over Individuals (Original)")
plot_calibration_stability(pred_sta, y, BS_number, "Calibration stability plot (Original)")
instability_df = calculate_classification_instability(pred_sta,0.100, BS_number)
plot_classification_instability(instability_df)

# SMOTENC Model
# prob_original_smnc = pd.DataFrame(model_smnc.predict_proba(x)[:, 1], columns=['prob_original'])
# c_sta_smnc, pred_sta_smnc = prediction_stability(model_smnc, BS_number, x, y, prob_original_smnc)
# plot_prediction_instability(pred_sta_smnc, BS_number, "Prediction Instability (SMOTENC)")
# plot_mape(pred_sta_smnc, BS_number, "MAPE over Individuals (SMOTENC)")
# plot_calibration_stability(pred_sta_smnc, y, BS_number, "Calibration stability plot (SMOTENC)")
# instability_df_smnc = calculate_classification_instability(pred_sta_smnc,0.100, BS_number)
# plot_classification_instability(instability_df_smnc)

# # BorderlineSMOTE Model
# prob_original_bor = pd.DataFrame(model_bor.predict_proba(x)[:, 1], columns=['prob_original'])
# c_sta_bor, pred_sta_bor = prediction_stability(model_bor, BS_number, x, y, prob_original_bor)
# plot_prediction_instability(pred_sta_bor, BS_number, "Prediction Instability (BorderlineSMOTE)")
# plot_mape(pred_sta_bor, BS_number, "MAPE over Individuals (BorderlineSMOTE)")
# plot_calibration_stability(pred_sta_bor, y, BS_number, "Calibration stability plot (BorderlineSMOTE)")
# instability_df_bor = calculate_classification_instability(pred_sta_bor,0.100, BS_number)
# plot_classification_instability(instability_df_bor)


# ADASYN Model
# prob_original_ada = pd.DataFrame(model_ada.predict_proba(x)[:, 1], columns=['prob_original'])
# c_sta_ada, pred_sta_ada = prediction_stability(model_ada, BS_number, x, y, prob_original_ada)
# plot_prediction_instability(pred_sta_ada, BS_number, "Prediction Instability (ADASYN)")
# plot_mape(pred_sta_ada, BS_number, "MAPE over Individuals (ADASYN)")
# plot_calibration_stability(pred_sta_ada, y, BS_number, "Calibration stability plot (ADASYN)")
# instability_df_ada = calculate_classification_instability(pred_sta_ada,0.100, BS_number)
# plot_classification_instability(instability_df_ada)



#Small
try:
    df = pd.read_csv("/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Academic Desk/2024Instability/model_instability/dataset/sampled_gusto_dataset(Sheet1).csv")
except FileNotFoundError:
    print("Error: The specified data file was not found. Please check the file path.")
    exit()

df['sex'] = df['sex'].apply(lambda x: 1 if x == 'male' else 0)
df['pmi'] = df['pmi'].apply(lambda x: 1 if x == 'yes' else 0)

# Keep the outcome (day30) and the 7 predictors of interest
x = df[["age", "ste", "htn", "sex", "hyp", "hrt", "pmi","sysbp"]]
y = df["day30"]

results = '/home/natthanaphop.isa/model_instability/results/instability2/full'
os.makedirs(results, exist_ok=True)

c_bs = c_statistic_optimism(model, BS_number, x, y)
mean_c_bs = statistics.mean(c_bs)
sd_c_bs = statistics.stdev(c_bs)
ci_c_bs = sms.DescrStatsW(c_bs).tconfint_mean()
print(f"Original model C-statistic optimism - Mean: {mean_c_bs:.4f}, SD: {sd_c_bs:.4f}, 95%CI: {ci_c_bs[0]:.4f} - {ci_c_bs[1]:.4f}")
plot_histogram_with_density(c_bs, "C-statistic optimism (Original)")

prob_original = pd.DataFrame(model.predict_proba(x)[:, 1], columns=['prob_original'])
c_sta, pred_sta = prediction_stability(model, BS_number, x, y, prob_original)
plot_prediction_instability(pred_sta, BS_number, "Prediction Instability (Original)")
plot_mape(pred_sta, BS_number, "MAPE over Individuals (Original)")
plot_calibration_stability(pred_sta, y, BS_number, "Calibration stability plot (Original)")
instability_df = calculate_classification_instability(pred_sta,0.100, BS_number)
plot_classification_instability(instability_df)

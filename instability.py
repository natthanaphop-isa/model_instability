import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import roc_auc_score, mean_squared_error, roc_curve
from scipy.stats import norm, multivariate_normal
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Function to generate data
def generate_data_1_predictor(N, n_true_predictors, n_noise_predictors, cor0=0, cor1=0, beta_0=0, var0=4):
    n_predictors = n_true_predictors + n_noise_predictors
    mean = np.zeros(n_predictors)
    cov = np.eye(n_predictors) * cor1
    cov[:n_true_predictors, :n_true_predictors] = cor0
    cov = cov + np.diag([var0] + [1] * n_noise_predictors)
    X = multivariate_normal(mean, cov).rvs(size=N)
    
    beta = np.array([1] + [0] * n_noise_predictors)
    logits = beta_0 + np.dot(X, beta)
    prob = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(N) < prob).astype(int)
    data = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_predictors)])
    data['y'] = y
    return data

# Logistic regression model validation
def validate_model(data, model):
    X = data.drop(columns=['y']).values
    y = data['y'].values
    pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, pred)
    mse = mean_squared_error(y, pred)
    return {"AUC": auc, "MSE": mse}

# Shrinkage recalibration
def recalibrate_model(shrinkage_factor, model, data):
    X = data.drop(columns=['y']).values
    y = data['y'].values
    coeffs = model.coef_ * shrinkage_factor
    intercept = model.intercept_
    model_shrunk = LogisticRegression()
    model_shrunk.coef_ = coeffs
    model_shrunk.intercept_ = intercept
    model_shrunk.fit(X, y)  # Refit with shrunk coefficients
    return model_shrunk

# Elastic net model
def elastic_net_model(data, alpha=1.0):
    X = data.drop(columns=['y']).values
    y = data['y'].values
    model = LogisticRegressionCV(cv=5, penalty='elasticnet', l1_ratios=[alpha], solver='saga', scoring='roc_auc', max_iter=10000)
    model.fit(X, y)
    return model

# Generate data
N = 1000
n_true_predictors = 1
n_noise_predictors = 10
data = generate_data_1_predictor(N, n_true_predictors, n_noise_predictors)

# Fit full logistic regression model
X = data.drop(columns=['y']).values
y = data['y'].values
full_model = LogisticRegression()
full_model.fit(X, y)
full_model_perf = validate_model(data, full_model)

# Fit elastic net model
elastic_net = elastic_net_model(data, alpha=0.5)
elastic_net_perf = validate_model(data, elastic_net)

# Output performance
print("Full Model Performance:", full_model_perf)
print("Elastic Net Performance:", elastic_net_perf)

# Calibration plot
def calibration_plot(model, data):
    X = data.drop(columns=['y']).values
    y = data['y'].values
    preds = model.predict_proba(X)[:, 1]
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(preds, y, "o", alpha=0.1)
    plt.xlabel("Estimated Risk")
    plt.ylabel("Observed")
    plt.title("Calibration Plot")
    plt.show()

calibration_plot(full_model, data)
calibration_plot(elastic_net, data)


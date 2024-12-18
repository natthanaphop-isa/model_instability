import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Load dataset
df = pd.read_excel('/Users/natthanaphop_isa/Library/CloudStorage/GoogleDrive-natthanaphop.isa@gmail.com/My Drive/Academic Desk/2024Instability/model_instability/dataset/lp_internal_dataset.xlsx')
list_of_col = ['age', 'sex', 'HT', 'lipid', 'BMI', 'waistcir', 'calfcir', 'exhaustion']
target_column = 'frail'

# Prepare features and target
X = df[list_of_col]
y = df[target_column]

# Train the original model
original_model = LogisticRegression(C=2, class_weight='balanced', penalty='l2', solver='liblinear', max_iter=1000)
original_model.fit(X, y)
original_probs = original_model.predict_proba(X)[:, 1]

# Bootstrap settings
n_bootstrap = 50
bootstrap_models = []
bootstrap_probs = []

# Bootstrap the dataset and train models using GridSearchCV
for i in range(n_bootstrap):
    # Resample the dataset
    boot_df = resample(df, n_samples=len(df), random_state=i)
    X_boot = boot_df[list_of_col]
    y_boot = boot_df[target_column]
    
    # Train logistic regression using GridSearchCV
    logistic_model = LogisticRegression()
    param_grid = { 
        'penalty': ['l1', 'l2'],  
        'C': np.arange(1, 2, 0.5), 
        'class_weight': ['balanced', None], 
        'solver': ['liblinear', 'saga']  
    }
    grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=2)
    grid_search.fit(X_boot, y_boot)
    
    # Get the best model and train it
    best_params = grid_search.best_params_
    print(f"Bootstrap {i+1} - Best Parameters: {best_params}")
    
    best_model = LogisticRegression(**best_params)
    best_model.fit(X_boot, y_boot)
    
    # Store the trained model and its predicted probabilities
    bootstrap_models.append(best_model)
    bootstrap_probs.append(best_model.predict_proba(X)[:, 1])  # Predictions on the same dataset as the original model
    
# Rows correspond to bootstrap iterations, and columns correspond to data points (individuals)
bootstrap_probs_array = np.array(bootstrap_probs)

# Calculate percentiles for each individual (column-wise operation)
percentile_2_5 = np.percentile(bootstrap_probs_array, 2.5, axis=0)
percentile_97_5 = np.percentile(bootstrap_probs_array, 97.5, axis=0)

# Calculate LOWESS lines for 2.5th and 97.5th percentiles
lowess_2_5 = sm.nonparametric.lowess(percentile_2_5, original_probs, frac=0.3)  # Adjust frac for smoothness
lowess_97_5 = sm.nonparametric.lowess(percentile_97_5, original_probs, frac=0.3)

# Plot comparison of predicted probabilities
plt.figure(figsize=(10, 6))

# Scatter plots for each bootstrapped model
for i, probs in enumerate(bootstrap_probs):
    plt.scatter(original_probs, probs, label=f"Bootstrap Model {i+1}", alpha=0.6, s=2)

# Add ideal line for reference
plt.plot([0, 1], [0, 1], 'k-', lw=1, label="Ideal Line")

# Add 2.5th and 97.5th percentile lines
# plt.scatter(original_probs, percentile_2_5, label="2.5th Percentile Line")
# plt.scatter(original_probs, percentile_97_5, label="97.5th Percentile Line")
plt.plot(lowess_2_5[:, 0], lowess_2_5[:, 1], 'k--', lw=1, label="LOWESS 2.5th Percentile Line")
plt.plot(lowess_97_5[:, 0], lowess_97_5[:, 1], 'k--', lw=1, label="LOWESS 97.5th Percentile Line")

# Plot settings
plt.xlabel("Original Model Predicted Probabilities")
plt.ylabel("Bootstrapped Models Predicted Probabilities")
plt.title(f"Comparison of Predicted Probabilities: Original Model vs. {n_bootstrap} Bootstrapped Models")
#plt.legend()
plt.grid(True)
plt.show()

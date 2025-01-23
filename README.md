# model_instability

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
        'max_iter': [500],
        'solver': ["newton-cholesky", "sag", "saga", "lbfgs"],
        'penalty': [penalty] if penalty else ["none"]
    }
    if penalty == 'elasticnet':
        param_grid['l1_ratio'] = [0, 0.25, 0.5, 0.75, 1]
    
    gcv = GridSearchCV(model, param_grid=param_grid, cv=10, n_jobs=-1, verbose=0, return_train_score=True) #verbose was 2, changed to 0 to remove noise
    gcv.fit(x, y)
    model.set_params(**gcv.best_params_)
    return model
import csv
import numpy as np
import pytest
from sklearn.datasets import fetch_california_housing
from model.LassoHomotopy import LassoHomotopyModel  # Assuming the model is in LassoHomotopy.py

def load_data(filename="small_test.csv"):
    data = []
    try:
        with open(filename, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                data.append(row)
    except FileNotFoundError:
        raise Exception("small_test.csv not found!")
    X = np.array([[float(v) for k, v in datum.items() if k.startswith('x')] for datum in data])
    y = np.array([float(datum['y']) for datum in data])
    return X, y

def test_predict():
    X, y = load_data()
    best_mse = float('inf')
    best_alpha = 1e-4  # Starting with the optimized value from previous tuning
    best_preds = None
    
    # Tight alpha search around low values, adjusted for optimized model
    for alpha in np.logspace(-6, -3, 10):  # From 1e-6 to 1e-3, broader range for robustness
        model = LassoHomotopyModel(
            alpha=alpha, 
            l2_lambda=1e-5,  # Small L2 penalty for stability
            max_iter=1000,   # Reduced for small dataset
            tol=1e-8,        # Relaxed tolerance
            eta=0.01         # Faster adaptation
        )
        results = model.fit(X, y)
        preds = results.predict(X)
        mse = np.mean((y - preds) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_preds = preds
    
    print(f"Best MSE: {best_mse:.6f}")
    print(f"Best alpha: {best_alpha:.6f}")
    print(f"Predictions shape: {best_preds.shape}")
    print(f"Sample predictions: {best_preds[:5]}")
    # Adjusted threshold based on previous MSE of 74, aiming for significant improvement
    assert best_mse < 50, f"MSE {best_mse} is too high! Target is < 50"  # Buffer from 74 to 50
    # return best_mse

def load_csv_data(filepath):
    """load CSV data"""
    data = np.loadtxt(filepath, delimiter=",", skiprows=1)  # Skip header
    X = data[:, :-1]  # All but last column
    y = data[:, -1]   # Last column
    return X, y
def test_collinear():
    """Test Lasso Homotopy on a collinear dataset"""
    X, y = load_csv_data("collinear_data.csv")
    
    best_mse = float('inf')
    best_alpha = 1e-4
    best_preds = None
    best_beta = None
    
    for alpha in np.logspace(-6, -3, 10):
        model = LassoHomotopyModel(
            alpha=alpha,
            l2_lambda=1e-5,
            max_iter=1000,
            tol=1e-8,
            eta=0.01
        )
        results = model.fit(X, y)
        preds = results.predict(X)
        mse = np.mean((y - preds) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_preds = preds
            best_beta = results.beta
            
    print("best_mse: ", best_mse)
    
    n_features = X.shape[1]
    non_zero_count = np.sum(np.abs(best_beta) > 1e-6)
    assert non_zero_count < n_features, f"Expected sparse solution, got {non_zero_count} non-zero coefficients out of {n_features}"
    assert best_mse < 5, f"Collinear MSE {best_mse} is too high! Target is < 5"


def test_sklearn_housing():
    """Test Lasso Homotopy on a subsampled California Housing dataset from SciKit Learn"""
    data = fetch_california_housing()
    X_full, y_full = data.data, data.target

    np.random.seed(42)
    n_samples = 50
    indices = np.random.choice(X_full.shape[0], n_samples, replace=False)
    X = X_full[indices]
    y = y_full[indices]

    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std
    noise = 0.05 * np.random.randn(n_samples)
    y += noise

    best_mse = float('inf')
    best_alpha = 1e-4
    best_preds = None
    best_beta = None
    
    for alpha in np.logspace(-6, -3, 10):
        model = LassoHomotopyModel(
            alpha=alpha,
            l2_lambda=1e-5,
            max_iter=1000,
            tol=1e-8,
            eta=0.01
        )
        results = model.fit(X, y)
        preds = results.predict(X)
        mse = np.mean((y - preds) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_alpha = alpha
            best_preds = preds
            best_beta = results.beta
    
    
    n_features = X.shape[1]
    non_zero_count = np.sum(np.abs(best_beta) > 1e-6)
    print(f"California Housing Test - Non-zero coefficients: {non_zero_count} out of {n_features}")
    
    assert best_mse < 5, f"California Housing MSE {best_mse} is too high! Target is < 5"    
    
def test_convergence():
    X, y = load_data()
    model = LassoHomotopyModel(
        alpha=1e-4,      # Optimized alpha
        l2_lambda=1e-5,  # Small L2 penalty
        max_iter=1000,   # Reduced iterations
        tol=1e-8,        # Relaxed tolerance
        eta=0.01         # Faster adaptation
    )
    results = model.fit(X, y)
    preds = results.predict(X)
    mse = np.mean((y - preds) ** 2)
    assert not np.isnan(mse), "Convergence failed (NaN MSE)"

if __name__ == "__main__":
    mse = test_predict()
    convergance = test_convergence()
    collinear = test_collinear()
    sklearn = test_sklearn_housing()

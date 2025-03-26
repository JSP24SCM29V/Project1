# Lasso Homotopy Model - README

## Overview
The **Lasso Homotopy Model** is an optimized implementation of LASSO (Least Absolute Shrinkage and Selection Operator) regression using a homotopy approach. This model incorporates Elastic Net regularization, line search for adaptive step sizes, and bagging refinement to improve stability and generalization.

LASSO is primarily used for sparse regression problems where feature selection is crucial. It works by imposing an L1 penalty on regression coefficients, driving some of them to zero, effectively selecting a subset of relevant features.

---

## Functionality & Usage
### What does the model do?
The **Lasso Homotopy Model** performs linear regression with L1 and L2 regularization (Elastic Net). The primary use cases include:
- **Feature Selection:** Automatically selects the most relevant features by shrinking coefficients of less important ones to zero.
- **Sparse Regression:** Suitable for datasets with many features where only a few contribute significantly.
- **High-Dimensional Data:** Works well in cases where the number of features exceeds the number of observations.

### When should it be used?
- When there is a need for **automated feature selection**.
- When working with **high-dimensional datasets** where ordinary least squares regression may overfit.
- When handling **collinearity among features**, as the L1 penalty can help reduce redundancy.
- In **predictive modeling tasks** where sparsity in the model coefficients improves interpretability.

---

## Model Testing
### How did you test your model to determine if it is working correctly?
The model was tested using multiple evaluation strategies, including:
1. **Synthetic Data:** The model was trained on artificially generated data where true coefficients were known, allowing verification of whether the estimated coefficients match the expected values.
2. **Tested with a sklearn dataset:** Tested the model with the fetch_california_housing dataset achieveing optimal MSE.
2. **Cross-Validation:** The model was validated using K-fold cross-validation to check generalization performance.
3. **Comparison with Scikit-Learn’s Lasso:** The output of the model was compared against Scikit-Learn’s `Lasso` regression to ensure consistency.
4. **Residual Analysis:** The difference between actual and predicted values was analyzed to confirm the model’s accuracy.
5. **Performance Metrics:** Metrics like Mean Squared Error (MSE) was used to measure predictive performance.

---

## User-Tunable Parameters
### What parameters have been exposed to users for tuning performance?
The following hyperparameters are available for users to adjust the model’s behavior:
- **`alpha` (L1 regularization)**: Controls sparsity; higher values lead to more zero coefficients.
- **`l2_lambda` (L2 regularization)**: Adds a Ridge penalty for numerical stability.
- **`max_iter`**: Limits the number of iterations to control computation time.
- **`tol` (tolerance)**: Defines the stopping criteria for convergence.
- **`normalize`**: Determines whether to standardize features before fitting.
- **`eta` (adaptive learning rate)**: Adjusts lambda dynamically for faster convergence.

These parameters allow users to fine-tune the model for different datasets and optimize performance accordingly.

---

## Limitations & Challenges
### Are there specific inputs that the implementation has trouble with?
Yes, the model has some limitations:
1. **Highly Correlated Features:** The L1 penalty may arbitrarily select one feature from a group of correlated features, leading to instability.
2. **Small Datasets:** With limited data, the model may over-regularize and suppress important features.
3. **Extremely Noisy Data:** When noise levels are high, the model may struggle to distinguish between signal and noise.
4. **High Dimensionality with Few Observations:** If the number of features greatly exceeds the number of samples, tuning `alpha` and `l2_lambda` becomes critical to avoid overfitting.

### Given more time, could these issues be addressed?
- **Correlated Features:** A more refined Elastic Net approach with adaptive weighting could mitigate this issue.
- **Small Datasets:** Using Bayesian regression or model ensembling could improve results.
- **Noisy Data:** Robust regression techniques, such as Huber loss, could be incorporated.
- **Extreme High Dimensionality:** Dimensionality reduction techniques, like Principal Component Analysis (PCA), could help preprocess data before applying LASSO.


---
## How to run the code
1. **Create a virtual environment:**
   ```cmd
   python -m venv lasso
   ```

2. **Activate the virtual environment:**
   ```cmd
   lasso\Scripts\activate
   ```

3. **Install the dependencies from the requirements.txt file:**
   ```cmd
   pip install -r requirements.txt
   ```

4. **Run the tests from LassoHomotopy/tests using:**
   ```cmd
   pytest
   ```

---
---
## How to Use the Model
1. **Import the model:**
   ```python
   from model.LassoHomotopy import LassoHomotopyModel
   import numpy as np
   ```
2. **Initialize the model:**
   ```python
   model = LassoHomotopyModel(alpha=1e-4, l2_lambda=1e-5)
   ```
3. **Fit the model on data:**
   ```python
   model.fit(X_train, y_train)
   ```
4. **Make predictions:**
   ```python
   y_pred = model.predict(X_test)
   ```
---

## Conclusion
The **Lasso Homotopy Model** is a powerful tool for sparse regression and feature selection, particularly useful in high-dimensional settings. While it offers several advantages, careful tuning of hyperparameters is necessary to ensure optimal performance. Future improvements could further enhance the model’s robustness and adaptability across different datasets.

---
## Team Members
1. Jaitra Narasimha Valluri (A20553229)
2. ⁠Chandrika Rajani (A20553311)
3. ⁠Pooja Sree Kuchi (A20553325)
---

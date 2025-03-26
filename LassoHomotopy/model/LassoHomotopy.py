import numpy as np
from scipy import linalg

class LassoHomotopyModel:
    def __init__(self, alpha=1e-4, l2_lambda=1e-5, max_iter=1000, tol=1e-8, normalize=True, eta=0.01):
        """
        Optimized LASSO Homotopy with Elastic Net and Line Search
        alpha: L1 regularization (tuned for small dataset)
        l2_lambda: L2 regularization for stability (small Ridge penalty)
        max_iter: reduced for small dataset
        tol: relaxed for faster convergence
        normalize: center and scale features
        eta: adaptive lambda learning rate (increased for faster adaptation)
        """
        self.alpha = alpha
        self.l2_lambda = l2_lambda  # Small L2 penalty for stability
        self.max_iter = max_iter
        self.tol = tol
        self.normalize = normalize
        self.eta = eta
        self.beta = None
        self.active_set = None
        self.X_mean = None
        self.X_std = None
    
    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape
        
        # Normalize features
        if self.normalize:
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_std[self.X_std < self.tol] = 1.0
            X = (X - self.X_mean) / self.X_std
        
        # Initialize
        self.beta = np.zeros(n_features)
        self.active_set = set()
        lambda_curr = np.abs(X.T.dot(y)).max() / n_samples
        lambda_target = self.alpha
        
        # Step 1: Regularization path with Elastic Net
        self._compute_regularization_path(X, y, lambda_curr, lambda_target)
        
        # Step 2: Refine with adaptive lambda and bagging
        self._refine_solution(X, y, n_samples)
        
        # Simple bagging: average over perturbed solutions
        self._bagging_refinement(X, y, n_samples)
        
        return LassoHomotopyResults(self.beta, self.X_mean, self.X_std)
    
    def _compute_regularization_path(self, X, y, lambda_start, lambda_end):
        residual = y - X.dot(self.beta)
        n_samples, n_features = X.shape
        
        for _ in range(self.max_iter):
            # Include L2 penalty in gradient
            correlations = X.T.dot(residual) / n_samples - 2 * self.l2_lambda * self.beta
            c_max = np.abs(correlations).max()
            
            if c_max < self.tol or lambda_start <= lambda_end + self.tol:
                break
                
            new_active = np.where(np.abs(correlations) >= c_max - self.tol)[0]
            self.active_set.update(new_active)
            
            X_active = X[:, list(self.active_set)]
            G = X_active.T.dot(X_active) / n_samples + 2 * self.l2_lambda * np.eye(len(self.active_set))
            signs = np.sign(correlations[list(self.active_set)])
            A = np.ones(len(self.active_set))
            
            try:
                direction = linalg.solve(G, A, assume_a='pos')
                direction = direction * signs / np.sqrt(A.dot(direction))
            except linalg.LinAlgError:
                break
            
            beta_direction = np.zeros(n_features)
            active_idx = list(self.active_set)
            beta_direction[active_idx] = direction
            
            gamma = self._compute_step_size(X, y, residual, beta_direction, lambda_start)
            if gamma < self.tol:
                break
                
            self.beta += gamma * beta_direction
            residual = y - X.dot(self.beta)
            lambda_start -= gamma * c_max
            
            self.beta[np.abs(self.beta) < self.tol] = 0
            self.active_set = set(np.where(np.abs(self.beta) > self.tol)[0])
    
    def _compute_step_size(self, X, y, residual, direction, lambda_curr):
        """Armijo line search for step size"""
        n_samples = X.shape[0]
        beta_new = self.beta + direction
        obj_curr = (0.5 * np.sum(residual**2) / n_samples + 
                    self.alpha * np.sum(np.abs(self.beta)) + 
                    self.l2_lambda * np.sum(self.beta**2))
        
        gamma = 1.0
        sigma = 0.1  # Armijo condition parameter
        for _ in range(10):  # Limit iterations
            beta_test = self.beta + gamma * direction
            residual_test = y - X.dot(beta_test)
            obj_new = (0.5 * np.sum(residual_test**2) / n_samples + 
                       self.alpha * np.sum(np.abs(beta_test)) + 
                       self.l2_lambda * np.sum(beta_test**2))
            
            if obj_new <= obj_curr - sigma * gamma * np.sum(direction**2):
                break
            gamma *= 0.5
        
        return gamma if gamma > self.tol else self.tol
    
    def _refine_solution(self, X, y, n_samples):
        lambda_n = self.alpha
        for _ in range(min(20, n_samples)):  # Fewer iterations
            residual = y - X.dot(self.beta)
            active_idx = list(self.active_set)
            if not active_idx:
                break
                
            X_active = X[:, active_idx]
            G = X_active.T.dot(X_active) / n_samples + 2 * self.l2_lambda * np.eye(len(active_idx))
            signs = np.sign(self.beta[active_idx])
            
            pred_error = np.mean(residual)
            grad = 2 * n_samples * X_active.mean(axis=0).T.dot(linalg.inv(G)).dot(signs) * pred_error
            lambda_n *= np.exp(-self.eta * grad)
            lambda_n = max(1e-10, min(lambda_n, 0.005))  # Adjusted bounds
            
            self._compute_regularization_path(X, y, lambda_n + 1e-6, lambda_n)
    
    def _bagging_refinement(self, X, y, n_samples, n_bags=5):
        """Simple bagging: average solutions over perturbed datasets"""
        beta_avg = np.zeros_like(self.beta)
        for _ in range(n_bags):
            # Bootstrap sample
            idx = np.random.choice(n_samples, n_samples, replace=True)
            X_bag = X[idx]
            y_bag = y[idx]
            
            # Fit on bootstrap sample
            beta_bag = np.zeros_like(self.beta)
            active_set_bag = set()
            lambda_curr = np.abs(X_bag.T.dot(y_bag)).max() / n_samples
            residual = y_bag - X_bag.dot(beta_bag)
            
            for _ in range(self.max_iter):
                correlations = X_bag.T.dot(residual) / n_samples - 2 * self.l2_lambda * beta_bag
                c_max = np.abs(correlations).max()
                if c_max < self.tol:
                    break
                new_active = np.where(np.abs(correlations) >= c_max - self.tol)[0]
                active_set_bag.update(new_active)
                
                X_active = X_bag[:, list(active_set_bag)]
                G = X_active.T.dot(X_active) / n_samples + 2 * self.l2_lambda * np.eye(len(active_set_bag))
                signs = np.sign(correlations[list(active_set_bag)])
                A = np.ones(len(active_set_bag))
                
                try:
                    direction = linalg.solve(G, A, assume_a='pos')
                    direction = direction * signs / np.sqrt(A.dot(direction))
                except linalg.LinAlgError:
                    break
                
                beta_direction = np.zeros_like(beta_bag)
                active_idx = list(active_set_bag)
                beta_direction[active_idx] = direction
                gamma = self._compute_step_size(X_bag, y_bag, residual, beta_direction, lambda_curr)
                
                beta_bag += gamma * beta_direction
                residual = y_bag - X_bag.dot(beta_bag)
                lambda_curr -= gamma * c_max
                
                beta_bag[np.abs(beta_bag) < self.tol] = 0
                active_set_bag = set(np.where(np.abs(beta_bag) > self.tol)[0])
            
            beta_avg += beta_bag
        
        self.beta = beta_avg / n_bags

class LassoHomotopyResults:
    def __init__(self, beta, X_mean, X_std):
        self.beta = beta
        self.X_mean = X_mean
        self.X_std = X_std
    
    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        if self.X_mean is not None and self.X_std is not None:
            X = (X - self.X_mean) / self.X_std
        return X.dot(self.beta)

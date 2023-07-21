import numpy as np


class PortfolioOptimization:
    def __init__(
        self,
        returns,
        cov_matrix,
        expected_return,
        initial_weights,
        learning_rate=0.01,
        tolerance=1e-6,
        max_iterations=1000,
    ):
        """
        Constructor for the PortfolioOptimization class.

        Parameters:
            returns (numpy array): 1D numpy array containing the expected returns of each asset.
            cov_matrix (numpy array): 2D numpy array containing the covariance matrix of assets.
            expected_return (float): The target expected return of the portfolio.
            initial_weights (numpy array): 1D numpy array containing the initial weights of assets.
            learning_rate (float, optional): Learning rate for the gradient descent optimization. Default is 0.01.
            tolerance (float, optional): Convergence tolerance for the optimization. Default is 1e-6.
            max_iterations (int, optional): Maximum number of iterations for the optimization. Default is 1000.
        """
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.expected_return = expected_return
        self.initial_weights = initial_weights
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def _objective_function(self, weights):
        """
        Internal method to calculate the portfolio risk (variance).

        Parameters:
            weights (numpy array): 1D numpy array containing the weights of assets.

        Returns:
            float: The portfolio risk (variance).
        """
        portfolio_risk = weights.T @ self.cov_matrix @ weights
        return portfolio_risk

    def _projection_onto_hyperplane(self, point, normal_vector, b):
        """
        Internal method to project a point onto the hyperplane defined by: <x, normal_vector> = b.

        Parameters:
            point (numpy array): The point to be projected onto the hyperplane.
            normal_vector (numpy array): The normal vector of the hyperplane.
            b (float): The offset parameter of the hyperplane.

        Returns:
            numpy array: The projected point onto the hyperplane.
        """
        return (
            point
            - (np.dot(point, normal_vector) - b)
            / np.dot(normal_vector, normal_vector)
            * normal_vector
        )

    def _projection_onto_halfspace(self, point, normal_vector, b):
        """
        Internal method to project a point onto the halfspace defined by: <x, normal_vector> >= b.

        Parameters:
            point (numpy array): The point to be projected onto the halfspace.
            normal_vector (numpy array): The normal vector of the halfspace.
            b (float): The offset parameter of the halfspace.

        Returns:
            numpy array: The projected point onto the halfspace.
        """
        if np.dot(point, normal_vector) >= b:
            return point
        else:
            return self._projection_onto_hyperplane(point, normal_vector, b)

    def _projection_onto_feasible_region(self, weights):
        """
        Internal method to project weights onto the feasible region of the portfolio.

        Parameters:
            weights (numpy array): 1D numpy array containing the weights of assets.

        Returns:
            numpy array: Projected weights onto the feasible region.
        """
        for _ in range(self.max_iterations):
            prev_weights = np.copy(weights)

            # Projection onto Constraint 1
            weights = self._projection_onto_hyperplane(weights, np.ones(weights.size), 1)

            # Projection onto Constraint 2
            weights = self._projection_onto_halfspace(
                weights, self.returns, self.expected_return
            )

            # Projection onto Constraint 3
            weights = np.maximum(weights, 0)

            # Convergence check
            if np.linalg.norm(weights - prev_weights) < self.tolerance:
                break

        return weights

    def optimize_portfolio(self):
        """
        Method to optimize the portfolio.

        Returns:
            numpy array: 1D numpy array containing the optimized weights of assets.
        """
        weights = self.initial_weights
        prev_weights = np.copy(weights)

        for _ in range(self.max_iterations):
            # Calculate the gradient
            gradient = self.cov_matrix @ weights

            # Update the weights using gradient descent
            weights = weights - self.learning_rate * gradient

            # Project onto the feasible region
            weights = self._projection_onto_feasible_region(weights)

            # Convergence check
            if np.linalg.norm(weights - prev_weights) < self.tolerance:
                break

            prev_weights = np.copy(weights)

        return weights

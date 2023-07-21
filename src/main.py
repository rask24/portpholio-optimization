import pickle
import numpy as np
import pandas as pd
import yaml

from plot import plot_optimal_portfolio_weights, plot_stock_prices
from PortfolioOptimization import PortfolioOptimization


def main():
    # Load the settings from the "setting.yml" file
    with open("./src/setting.yml", "r") as file:
        const: dict = yaml.safe_load(file)

    # Load the stock data from the "stock_data.pkl" file
    with open("./shared/stock_data.pkl", "rb") as file:
        df: pd.DataFrame = pickle.load(file)

    # Calculate the expected returns of each asset for the portfolio
    returns = df.mean().values

    # Calculate the covariance matrix of the assets for the portfolio
    cov_matrix = df.cov().values

    # Set the target expected return and initial weights for the portfolio optimization
    expected_return = const["opt"]["return"]
    initial_weights = np.array(const["opt"]["initial_weights"])

    # Create an instance of the PortfolioOptimization class
    optimizer = PortfolioOptimization(
        returns=returns,
        cov_matrix=cov_matrix,
        expected_return=expected_return,
        initial_weights=initial_weights,
        learning_rate=0.001,
        tolerance=1e-6,
        max_iterations=100000,
    )

    # Optimize the portfolio
    optimal_weights = optimizer.optimize_portfolio()

    # Display the optimal portfolio weights
    print("Optimal Portfolio Weights:")
    for weight, name in zip(optimal_weights, const["stock"].values()):
        print(f"{name}: {weight:.4f}")

    # Plot and save the stock prices in a time series graph
    plot_stock_prices(df, const)

    # Plot and save the optimal portfolio weights in a pie chart
    plot_optimal_portfolio_weights(optimal_weights, const)


if __name__ == "__main__":
    main()

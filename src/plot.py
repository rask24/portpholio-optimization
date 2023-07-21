import os

import matplotlib.pyplot as plt
import numpy as np


def plot_objective_values(objective_values):
    if not os.path.exists("./images"):
        os.makedirs("./images")

    fig = plt.figure()
    iterations = np.arange(objective_values.size)
    plt.plot(iterations, objective_values)
    plt.title("transition of objective values")
    plt.savefig("./images/transition.png")
    plt.close(fig)


def plot_stock_prices(df, const):
    if not os.path.exists("./images"):
        os.makedirs("./images")

    fig = plt.figure(figsize=(12, 8))
    for code, name in const["stock"].items():
        plt.plot(df.index, df[code].values, label=name)
    plt.legend()
    plt.title("Stock Prices")
    plt.savefig("./images/stock.png")
    plt.close(fig)  # Close the figure after saving


def plot_optimal_portfolio_weights(optimal_weights, const):
    if not os.path.exists("./images"):
        os.makedirs("./images")

    fig = plt.figure(figsize=(11, 8))
    labels = const["stock"].values()
    sizes = optimal_weights
    patches, _, _ = plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis("equal")  # Set aspect ratio to equal
    plt.title("Optimal Portfolio Weights")
    plt.legend(patches, labels, loc="best")
    plt.savefig("./images/results.png")
    plt.close(fig)  # Close the figure after saving

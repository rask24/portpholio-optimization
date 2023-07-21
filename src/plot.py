import matplotlib.pyplot as plt


def plot_stock_prices(df, const):
    fig = plt.figure(figsize=(12, 8))
    for code, name in const["stock"].items():
        plt.plot(df.index, df[code].values, label=name)
    plt.legend()
    plt.title("Stock Prices")
    plt.savefig("./images/stock.png")
    plt.close(fig)  # Close the figure after saving


def plot_optimal_portfolio_weights(optimal_weights, const):
    fig = plt.figure(figsize=(8, 8))
    labels = const["stock"].values()
    sizes = optimal_weights
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.axis("equal")  # Set aspect ratio to equal
    plt.title("Optimal Portfolio Weights")
    plt.savefig("./images/results.png")
    plt.close(fig)  # Close the figure after saving

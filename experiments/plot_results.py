import matplotlib.pyplot as plt

def plot_cost_history(history):
    plt.plot(history, label="MDL cost")
    plt.xlabel("Iteration")
    plt.ylabel("MDL cost")
    plt.title("Zmiana kosztu MDL w czasie")
    plt.legend()
    plt.show()

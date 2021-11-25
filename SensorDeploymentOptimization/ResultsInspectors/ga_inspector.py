import numpy as np

def plot_convergence(
                     results,
                     xlabel="Number of iterations $n$",
                     ylabel=r"Max objective value after $n$ iterations",
                     ax=None, name=None, alpha=0.2, yscale=None,
                     color=None, true_minimum=None,
                     **kwargs
                    ):
        
    losses = list(results)
    n_calls = len(losses)
    iterations = range(1, n_calls + 1)
    maxs = [np.max(losses[:i]) for i in iterations]
    min_maxs = min(maxs)
    cliped_losses = np.clip(losses, min_maxs, None)
    return plotter(iterations, maxs, cliped_losses, xlabel, ylabel, ax, name, alpha, yscale, color,
                            true_minimum, **kwargs)
    
def plotter(
            x, y1, y2,
            xlabel="Number of iterations $n$",
            ylabel=r"Max objective value after $n$ iterations",
            ax=None, name=None, alpha=0.2, yscale=None,
            color=None, true_minimum=None,
            **kwargs
           ):
    
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()

    ax.set_title("Convergence plot")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid()

    if yscale is not None:
        ax.set_yscale(yscale)

    ax.plot(x, y1, c=color, label=name, **kwargs)
    ax.scatter(x, y2, c=color, alpha=alpha)

    if true_minimum is not None:
        ax.axhline(true_minimum, linestyle="--",
                   color="r", lw=1,
                   label="True minimum")

    if true_minimum is not None or name is not None:
        ax.legend(loc="upper right")
        
    return ax

def read_files(directory):
    results = []
    
    import os
    import pickle
    for filename in os.listdir(directory):
        if not filename.startswith('.'):
            with open(os.path.join(directory, filename), 'rb') as f:
                results.append(pickle.load(f))

    return results
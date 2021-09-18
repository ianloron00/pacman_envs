from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
import matplotlib.pyplot as plt
import numpy as np

# https://matplotlib.org/stable/gallery/lines_bars_and_markers/fill_between_demo.html#sphx-glr-gallery-lines-bars-and-markers-fill-between-demo-py
# https://stable-baselines.readthedocs.io/en/master/_modules/stable_baselines/results_plotter.html#plot_results

def plot_training(dirs, num_timesteps, xaxis, title):
    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)

    xy_list = [ts2xy(timesteps_item, xaxis) for timesteps_item in tslist]
    # plot_curves(xy_list, xaxis, task_name)

    plt.figure(figsize=(8, 2))
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    
    EPISODES_WINDOW = 1000

    for (i, (x, y)) in enumerate(xy_list):
        
        plt.scatter(x, y, s=2)
        # Do not plot the smoothed curve at all if the timeseries is shorter than window size.
        if x.shape[0] >= EPISODES_WINDOW:
            # Compute and plot rolling mean with window of size EPISODE_WINDOW
            x, y_mean = results_plotter.window_func(x, y, EPISODES_WINDOW, np.mean)
            plt.plot(x, y_mean, color='darkblue')

            size = len(y_mean)
            _y = y[len(y) - size:]
            y_err = _y.std() * np.sqrt(1/len(_y) +
                          (_y - y.mean())**2 / np.sum((_y - y.mean())**2))

            y_var = 5 * y_err

            plt.fill_between(x, y_mean - y_var, y_mean + y_var, alpha=0.2)
            plt.fill_between(x, y_mean - y_err, y_mean + y_err, alpha=0.4)

    plt.xlim(minx, maxx)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel("Episode Rewards")
    plt.tight_layout()
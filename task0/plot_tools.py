import numpy as np
import matplotlib.pyplot as plt

plot_colors = ['#06d6a0', '#277da1', '#ee6c4d', '#5a189a', '#f3722c']


def plot(x_data, y_data, scatter: bool = False, color=None, label=None):
    if scatter:
        plt.scatter(x_data, y_data, color=color, label=label)
    else:
        plt.plot(x_data, y_data, color=color, label=label)
    if label is not None:
        plt.legend()


def x_label(label: str):
    plt.xlabel(label)


def y_label(label: str):
    plt.ylabel(label)


def plot_grid():
    plt.grid()


def plot_fill_under(x, y):
    plt.fill_between(x, y, step="pre", alpha=0.4)


def show_plot():
    plt.show()


def showPlots(plots):
    colors = ['#06d6a0', '#277da1', '#ee6c4d', '#5a189a', '#f3722c']
    plot_count = 0

    for i in range(len(plots)):
        if 'same' not in plots[i] or plots[i]['same'] == False:
            plot_count += 1

    fig, axs = plt.subplots(plot_count) if plot_count > 1 else plt
    pid = 0
    for i in range(len(plots)):
        if 'same' in plots[i] and plots[i]['same'] == True and pid > 0:
            pid -= 1
        if 'x' not in plots[i]:
            axs[pid].plot(plots[i]['data'], label=plots[i]['label'], color=colors[i % len(colors)])
        else:
            axs[pid].plot(plots[i]['x'][:len(plots[i]['data'])], plots[i]['data'], label=plots[i]['label'],
                          color=colors[i % len(colors)])
        axs[pid].grid()
        axs[pid].legend(loc='upper right')

        pid += 1

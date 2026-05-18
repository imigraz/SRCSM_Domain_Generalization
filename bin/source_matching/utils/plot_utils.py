import os
import numpy as np
import matplotlib.pyplot as plt
from bin.source_matching.utils.io_utils import create_dir


def plot_histogram(image, num_bins, title, color, output_path='', file_prefix=''):
    # caluculate histogram
    hist, bin_edges = np.histogram(image, num_bins)
    width = (bin_edges[1] - bin_edges[0])

    # plot histogram
    plt.bar(bin_edges[:-1], hist, align='center',
            width=width, edgecolor='k', facecolor=color, alpha=0.5)
    plt.title(title)

    out_filename = title.replace(' ', '_').lower()
    if file_prefix != '':
        out_filename = file_prefix + '_' + out_filename
    create_dir(output_path)
    plt.savefig(os.path.join(output_path, out_filename))
    plt.show()


def plot_histogram_hist_dict(hist_dict, num_bins, title, color, output_path='', file_prefix=''):
    # caluculate histogram

    key_min = min(list(hist_dict.keys()))
    key_max = max(list(hist_dict.keys()))
    step_size = (key_max - key_min) / (num_bins)
    bin_edges = np.arange(key_min, key_max + 1, step_size)

    hist = np.zeros(shape=[num_bins])
    keys_np = np.array(list(hist_dict.keys()))
    values_np = np.array(list(hist_dict.values()))

    for cur_bin in range(num_bins):
        bin_lower = bin_edges[cur_bin]
        bin_upper = bin_edges[cur_bin+1]
        if cur_bin != num_bins - 1:
            lower_tmp = np.where(keys_np >= bin_lower, True, False)
            upper_tmp = np.where(keys_np < bin_upper, True, False)
        else:
            lower_tmp = np.where(keys_np >= bin_lower, True, False)
            upper_tmp = np.where(keys_np <= bin_upper, True, False)
        hist[cur_bin] = np.sum(np.where(lower_tmp * upper_tmp, values_np, 0))

    # above is a workaround solution to get hist and bin_edges
    # hist, bin_edges = np.histogram(image, num_bins)
    width = (bin_edges[1] - bin_edges[0])

    # plot histogram
    plt.bar(bin_edges[:-1], hist, align='center',
            width=width, edgecolor='k', facecolor=color, alpha=0.5)
    plt.title(title)

    out_filename = title.replace(' ', '_').lower()
    if file_prefix != '':
        out_filename = file_prefix + '_' + out_filename
    create_dir(output_path)
    plt.savefig(os.path.join(output_path, out_filename))
    plt.show()


def plot_points(x_values, y_values, title, color=None, xlabel='', ylabel='', output_path='', file_prefix='', file_ext='.png'):
    plt.plot(x_values, y_values, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    out_filename = title.replace(' ', '_').lower()
    if file_prefix != '':
        out_filename = file_prefix + '_' + out_filename + file_ext
    create_dir(output_path)
    plt.savefig(os.path.join(output_path, out_filename))
    plt.show()


def plot_points_list(x_values_list, y_values_list, title, color_list=None, xlabel='', ylabel='', output_path='', file_prefix='', file_ext='.png'):
    if color_list is None:
        color_list = [None] * len(x_values_list)

    for x_values, y_values, color in zip(x_values_list, y_values_list, color_list):
        plt.plot(x_values, y_values, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    out_filename = title.replace(' ', '_').lower()
    if file_prefix != '':
        out_filename = file_prefix + '_' + out_filename + file_ext
    create_dir(output_path)
    plt.savefig(os.path.join(output_path, out_filename))
    plt.show()


def plot_points_list_and_points_mean(x_values_list, y_values_list, x_values_mean, y_values_mean, title, color_list=None, xlabel='', ylabel='', output_path='', file_prefix='', file_ext='.png'):
    if color_list is None:
        color_list = [None] * len(x_values_list)

    for x_values, y_values, color in zip(x_values_list, y_values_list, color_list):
        plt.plot(x_values, y_values, color=color)

    plt.plot(x_values_mean, y_values_mean, color='black', linewidth=2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    out_filename = title.replace(' ', '_').lower()
    if file_prefix != '':
        out_filename = file_prefix + '_' + out_filename + file_ext
    create_dir(output_path)
    plt.savefig(os.path.join(output_path, out_filename))
    plt.show()

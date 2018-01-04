import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
from matplotlib import cm
#
# # Have colormaps separated into categories:
# # http://matplotlib.org/examples/color/colormaps_reference.html
# cmaps = [('Perceptually Uniform Sequential', [
#             'viridis', 'plasma', 'inferno', 'magma']),
#          ('Sequential', [
#             'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
#             'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
#             'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
#          ('Sequential (2)', [
#             'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
#             'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
#             'hot', 'afmhot', 'gist_heat', 'copper']),
#          ('Diverging', [
#             'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
#             'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
#          ('Qualitative', [
#             'Pastel1', 'Pastel2', 'Paired', 'Accent',
#             'Dark2', 'Set1', 'Set2', 'Set3',
#             'tab10', 'tab20', 'tab20b', 'tab20c']),
#          ('Miscellaneous', [
#             'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
#             'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
#             'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]
#
#
# nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
# gradient = np.linspace(0, 1, 256)
# gradient = np.vstack((gradient, gradient))
#
#
# def plot_color_gradients(cmap_category, cmap_list, nrows):
#     fig, axes = plt.subplots(nrows=nrows)
#     fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
#     axes[0].set_title(cmap_category + ' colormaps', fontsize=14)
#
#     for ax, name in zip(axes, cmap_list):
#         ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
#         pos = list(ax.get_position().bounds)
#         x_text = pos[0] - 0.01
#         y_text = pos[1] + pos[3]/2.
#         fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)
#
#     # Turn off *all* ticks & spines, not just the ones with colormaps.
#     for ax in axes:
#         ax.set_axis_off()
#
#
# for cmap_category, cmap_list in cmaps:
#     plot_color_gradients(cmap_category, cmap_list, nrows)
#
# plt.show()


# from pylab import *
# cdict = {'red': ((0.0, 0.0, 0.0),
#                 (0.5, 1.0, 0.7),
#                 (1.0, 1.0, 1.0)),
#          'green': ((0.0, 0.0, 0.0),
#                  (0.5, 1.0, 0.0),
#                 (1.0, 1.0, 1.0)),
#          'blue': ((0.0, 0.0, 0.0),
#                   (0.5, 1.0, 0.0),
#                  (1.0, 0.5, 1.0))}
# my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)
# my_cmap = cm.get_cmap('gray_r', 100)
# print(type(np.random.rand(1, 10)))
# pcolor(np.random.rand(1, 10), cmap=my_cmap)
# colorbar()
# plt.show()

import matplotlib.pyplot as plt
from matplotlib.ticker import  MultipleLocator

from matplotlib.ticker import  FormatStrFormatter
import matplotlib as mpl
import numpy as np


def drawer(x, y):



    fig = plt.figure(facecolor='w')
    # 注意位置坐标，数字表示的是坐标的比例
    ax = fig.add_subplot(111)
    testInit = []
    for i in range(len(x)):

        testInit.append(x[i] * 100)

    test = []
    test.append(testInit)

    axim = plt.imshow(test, plt.get_cmap('gist_yarg'), interpolation='nearest')

    ax.set_xticklabels(y, rotation =40)
    plt.xticks(range(len(y)), y)
    plt.yticks([])

    # plt.colorbar(axim)

    plt.show()

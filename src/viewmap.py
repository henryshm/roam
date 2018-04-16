import math
import itertools

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Circle, ConnectionPatch

from sklearn.cluster import KMeans



class SimpleGraphProps:

    def __init__(self, size=80, color="#000000", alpha=0.5):
        self._size = size
        self._color = color
        self._alpha = alpha

    def get_size(self, i):
        return self._size

    def get_color(self, i):
        return self._color

    def get_label(self, i):
        return None

    def get_label_size(self, i):
        return 0


DEFAULT_PROPS = SimpleGraphProps()


def show_maps(maps, props=DEFAULT_PROPS, ncols=4, size=5):

    N = len(maps)

    if hasattr(props, '__iter__'):
        assert len(props) == N, "props must correspond to maps"
    else:
        props = [props] * N

    nrows = N // ncols
    if N % ncols != 0:
        nrows += 1
    if N < ncols:
        ncols = N

    plt.figure(figsize=(size*ncols, size*nrows))
    for i in range(N):
        ax = plt.subplot(nrows, ncols, i+1)
        show_map(ax, maps[i], props[i])

    plt.tight_layout()


def show_map(ax, amap, props):

    y = amap.y
    labels = amap.labels

    xvals = y[:,0]
    yvals = y[:,1]
    sizes = [props.get_size(l) for l in labels]
    colors = [props.get_color(l) for l in labels]

    # ensure axes are scaled equally and tick marks are suppressed
    ax.axis('equal')
    ax.set_title(amap.name, fontsize=20)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(plt.NullFormatter())

    # create all scatterplot points
    ax.scatter(xvals, yvals, color=colors, s=sizes, marker='o', alpha=0.5)

    # add labels as annotations
    for i, l in enumerate(labels):
        text = props.get_label(l)
        if text:
            ax.annotate(text, xy=(xvals[i], yvals[i]),
                horizontalalignment='center',
                verticalalignment='center',
                size=props.get_label_size(l),
            )

#     # legend
#     legend_patches = []
#     # patches for machine types
#     for (stype, color) in colormap.items():
#         patch = ax.scatter([], [], color=color, marker="o", s=getsize(medw), alpha=0.5, label=stype)
#         legend_patches.append(patch)
#     # size scale
#     for factor in [2,3,4,5]:
#         w = 10**factor
#         size=getsize(w)
#         patch = ax.scatter([], [], color="#707070", marker="o", s=size, alpha=0.5, label="{:,.0f} connections".format(w))
#         legend_patches.append(patch)

    #ax.legend(handles=legend_patches, labelspacing=5, handlelength=5, handletextpad=2., borderpad=3, ncol=2, loc=3)







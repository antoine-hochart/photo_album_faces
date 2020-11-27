import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from PIL import Image

######################################################################

FACES_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'faces')

######################################################################

def show_clusters(clusters, fnames, pseudos, figsize=(5,10), max_imgs=12):
    ncols = 12
    clusters_ = [
        np.random.choice(cluster, min(max_imgs, len(cluster)), replace=False)
        for cluster in clusters
        ]
    clusters_.append([cluster[0] for cluster in clusters if len(cluster) == 1])
    pseudos_ = pseudos + ['Snowgies']
    nrows = 0
    cluster_rows = []
    for cluster in clusters_:
        if len(cluster) > 1:
            n = int(np.ceil(len(cluster) / ncols))
            cluster_rows.append(range(nrows, nrows+n))
            nrows += n
        else:
            cluster_rows.append([])

    fig, axs = plt.subplots(nrows, ncols, figsize=figsize,
                            gridspec_kw={'wspace':0, 'hspace':0.1}, squeeze=False)

    for cluster, rows, pseudo in zip(clusters_, cluster_rows, pseudos_):
        if len(cluster) > 1:
            axs[rows[0],0].set_ylabel(pseudo.replace(' ', '\n'), fontsize=10,
                                      rotation='horizontal', ha='right', va='center')
            for j, ax in enumerate(axs[rows].reshape(-1)):
                if j < len(cluster):
                    idx = cluster[j]
                    img = Image.open(os.path.join(FACES_PATH, fnames[idx]))
                    ax.imshow(img, aspect='auto')
                    ax.axes.xaxis.set_ticks([])
                    ax.axes.yaxis.set_ticks([])
                else:
                    ax.axis('off')
    plt.show()

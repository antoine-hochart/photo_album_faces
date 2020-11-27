import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm

from pkg.gdl import GDL

from test.metrics import get_nmi, get_clustering_error
from test.utils import load_lfw_embeddings

#######################################################################

scores = {'nmi': [], 'err': []}

for seed in range(15):
    print("SEED {}".format(seed))
    print()

    ## # LARGE ALBUM 
    ## X, clusters_true = load_lfw_embeddings(n_familiars=7, n_strangers=14,
    ##     range_familiars=(50, 200), range_strangers=(1,3), seed=seed)
    ## param_range = range(5, 105, 5)

    # SMALL ALBUM 
    X, clusters_true = load_lfw_embeddings(n_familiars=40, n_strangers=10,
        range_familiars=(10, 20), range_strangers=(1,2), seed=seed)
    param_range = range(2, 52, 2)

    n_clusters = len(clusters_true)

    print()
    print("Parameter optimization (K)")
    nmi, err, times = [], [], []
    for nn in tqdm(param_range, ascii=True):
        t0 = time()
        gdl = GDL(n_neighbors=nn, n_clusters=n_clusters)
        gdl.fit(X)
        times.append(time() - t0)
        nmi.append(get_nmi(gdl.clusters, clusters_true))
        err.append(get_clustering_error(gdl.clusters, clusters_true))
    scores['nmi'].append(nmi)
    scores['err'].append(err)
    print("Average clustering run-time: {:.2f}s".format(np.mean(times)))
    print()

# score statistics
nmi_mean = np.mean(np.array(scores['nmi']), axis=0)
nmi_std = np.std(np.array(scores['nmi']), axis=0)
err_mean = np.mean(np.array(scores['err']), axis=0)
err_std = np.std(np.array(scores['err']), axis=0)
# nmi plot
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(param_range, nmi_mean, lw=2, ls='--', c='C0')
ax1.fill_between(param_range, nmi_mean + 1.96 * nmi_std, nmi_mean - 1.96 * nmi_std,
                 alpha=0.1, color='C0')
ax1.set_ylabel('Normalized Mutual Information', rotation=90)
ax1.set_ylim(0.2, 1)
ax1.set_xlabel('no. neighbors')
# clustering error plot
ax2 = ax1.twinx()
ax2.plot(param_range, err_mean, lw=2, ls='--', c='C1')
ax2.fill_between(param_range, err_mean + 1.96 * err_std, err_mean - 1.96 * err_std,
                 alpha=0.1, color='C1')
ax2.set_ylabel('Clustering error', rotation=90)
ax2.set_ylim(0, 0.8)

plt.show()
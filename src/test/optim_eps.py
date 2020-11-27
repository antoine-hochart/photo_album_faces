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

scores = {'nmi': [], 'err': [], 'n_clusters': []}

for seed in range(30, 60):
    print("SEED {}".format(seed))
    print()

    ## # LARGE ALBUM 
    ## X, clusters_true = load_lfw_embeddings(n_familiars=7, n_strangers=14,
    ##     range_familiars=(50, 200), range_strangers=(1,3), seed=seed)
    ## param_range = np.geomspace(0.001, 0.1, num=50)

    # SMALL ALBUM 
    X, clusters_true = load_lfw_embeddings(n_familiars=40, n_strangers=10,
        range_familiars=(10, 20), range_strangers=(1,2), seed=seed)
    param_range = np.geomspace(0.01, 0.70, num=50)

    print()
    print("Parameter optimization (eps)")
    nmi, err, ncl, times = [], [], [], []
    for eps in tqdm(param_range, ascii=True):
        t0 = time()
        ## gdl = GDL(n_neighbors=40, eps=eps)
        gdl = GDL(n_neighbors=12, eps=eps)
        gdl.fit(X)
        times.append(time() - t0)
        nmi.append(get_nmi(gdl.clusters, clusters_true))
        err.append(get_clustering_error(gdl.clusters, clusters_true))
        ncl.append(len(gdl.clusters))
    scores['nmi'].append(nmi)
    scores['err'].append(err)
    scores['n_clusters'].append(ncl)
    print("Average clustering run-time: {:.2f}s".format(np.mean(times)))
    print()

# score statistics
nmi_mean = np.mean(np.array(scores['nmi']), axis=0)
nmi_std = np.std(np.array(scores['nmi']), axis=0)
err_mean = np.mean(np.array(scores['err']), axis=0)
err_std = np.std(np.array(scores['err']), axis=0)
ncl_mean = np.mean(np.array(scores['n_clusters']), axis=0)
ncl_std = np.std(np.array(scores['n_clusters']), axis=0)
# nmi plot
_, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(param_range, nmi_mean, lw=2, ls='--', c='C0')
ax1.fill_between(param_range, nmi_mean + 1.96 * nmi_std, nmi_mean - 1.96 * nmi_std,
                 alpha=0.1, color='C0')
ax1.set_ylabel('Normalized Mutual Information', rotation=90)
ax1.set_ylim(0.90, 0.99)
ax1.set_xlabel('eps')
ax1.set_xscale('log')
# clustering error plot
ax2 = ax1.twinx()
ax2.plot(param_range, err_mean, lw=2, ls='--', c='C1')
ax2.fill_between(param_range, err_mean + 1.96 * err_std, err_mean - 1.96 * err_std,
                 alpha=0.1, color='C1')
ax2.set_ylabel('Clustering error', rotation=90)
ax2.set_ylim(0, 0.10)
# no. clusters
_, ax3 = plt.subplots(figsize=(10,5))
ax3.plot(param_range, ncl_mean, lw=2, ls='--', c='C0')
ax3.fill_between(param_range, ncl_mean + 1.96 * ncl_std, ncl_mean - 1.96 * ncl_std,
                 alpha=0.1, color='C0')
ax3.set_ylabel('No. clusters', rotation=90)
ax3.set_xlabel('eps')
ax3.set_xscale('log')

plt.show()
import sys
from os.path import dirname, join, abspath
sys.path.insert(0, abspath(join(dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt

from time import time
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

from pkg.gdl import GDL
from pkg.facenet import get_face_embedding

from test.metrics import get_clustering_error
from test.utils import load_lfw_embeddings, extract_lfw_faces

#######################################################################

scores = {'err': [], 'f1_score': []}

for seed in range(60, 90):
    print("SEED {}".format(seed))
    print()

    ## # LARGE ALBUM 
    ## X, clusters_true, names_in, names_out = load_lfw_embeddings(
    ##     n_familiars=7, n_strangers=14, range_familiars=(50, 200),
    ##     range_strangers=(1,3), seed=seed, return_all=True
    ##     )
    ## gdl = GDL(n_neighbors=40, eps=0.010)

    # SMALL ALBUM 
    X, clusters_true, names_in, names_out = load_lfw_embeddings(
        n_familiars=40, n_strangers=10, range_familiars=(10, 20),
        range_strangers=(1,2), seed=seed, return_all=True
        )
    gdl = GDL(n_neighbors=12, eps=0.13)

    print()
    print("Clustering")
    t0 = time()
    gdl.fit(X)
    print("Done ({:.2f}s)".format(time() - t0))
    cltr_err, mapping = get_clustering_error(gdl.clusters, clusters_true,
                                             return_map=True)
    print("Clustering error {:.5f}".format(cltr_err))

    print()
    print("Extracting and embedding test faces")
    t0 = time()
    faces_in, labels_in = extract_lfw_faces(names_in, extract_mode='test_in')
    faces_out, labels_out = extract_lfw_faces(names_out, extract_mode='test_out')
    faces_test = torch.cat((faces_in, faces_out), 0)
    labels_test = labels_in + labels_out

    X_test = get_face_embedding(faces_test, batch_size=1.0)
    print("Done ({:.2f}s)".format(time() - t0))

    print("No. of persons in test set: {}".format(
        len(np.unique(labels_in)) + len(labels_out)))
    print("No. of faces extracted from test set: "\
        "{} ({} in album / {} out of album)".format(
            len(faces_test), len(labels_in), len(labels_out)))

    print()
    print("Parameter estimation for prediction (alpha)")
    param_range = np.linspace(0.1, 5, num=100)
    f1, err, times = [], [], []
    for alpha in tqdm(param_range, ascii=True): 
        t0 = time()
        labels_pred = gdl.predict(X_test, alpha=alpha)
        times.append(time() - t0)
        labels_pred = [mapping[i] for i in labels_pred]
        y_true = [0 if i == -1 else 1 for i in labels_test]
        y_pred = [0 if i == -1 else 1 for i in labels_pred]
        f1.append(f1_score(y_true, y_pred))
        err.append(1 - accuracy_score(y_true, y_pred))
    scores['err'].append(err)
    scores['f1_score'].append(f1)
    print("Average prediction run-time: {:.3f}s".format(np.mean(times)))
    print()

# score statistics
f1_mean = np.mean(np.array(scores['f1_score']), axis=0)
f1_std = np.std(np.array(scores['f1_score']), axis=0)
# f1 score plot
fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(param_range, f1_mean, lw=2, ls='--', c='C0')
ax1.fill_between(param_range, f1_mean + 1.96 * f1_std, f1_mean - 1.96 * f1_std,
                 alpha=0.1, color='C0')
ax1.set_ylabel('F1 Score (in vs. out of album)', rotation=90)
## ax1.set_ylim(0.87, 0.99)
ax1.set_ylim(0.5, 1.0)
ax1.set_xlabel('alpha')
plt.show()
import os
import numpy as np
import torch

from time import time
from tqdm import tqdm
from PIL import Image
from torchvision.transforms.functional import to_tensor

from pkg.mtcnn import fixed_image_standardization
from pkg.facenet import get_face_embedding
from pkg.gdl import GDL
from pkg.viz import show_clusters
from pkg.pseudo import generate_pseudos

######################################################################

FACES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'faces')

######################################################################

def load_face_embeddings():
    """ Load face images from `faces` folder and apply FaceNet embedding. """
    fnames = os.listdir(FACES_PATH)

    faces = []
    print("Loading faces")
    for fname in tqdm(fnames, ascii=True):
        face = Image.open(os.path.join(FACES_PATH, fname))
        face = to_tensor(np.float32(face))
        face = fixed_image_standardization(face)
        faces.append(face)
    faces = torch.stack(faces)

    print("Computing face embeddings")
    embeddings = get_face_embedding(faces, batch_size=0.25)

    return embeddings, fnames

######################################################################

if __name__ == "__main__":
    print("Face embedding")
    t0 = time()
    faces, fnames = load_face_embeddings()
    print("Done ({:.2f}s)".format(time() - t0))

    print()
    print("Face clustering")
    t0 = time()
    gdl = GDL(n_neighbors=5, eps=0.050)
    gdl.fit(faces)
    print("Done ({:.2f}s)".format(time() - t0))

    print()
    n_clusters = len(gdl.clusters)
    print("No. clusters: {}".format(n_clusters))
    pseudos = generate_pseudos(n_clusters)
    show_clusters(gdl.clusters, fnames, pseudos, figsize=(9,9), max_imgs=12)
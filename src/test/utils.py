import os
import pandas as pd
import torch

from time import time
from PIL import Image

from pkg.mtcnn import MTCNN
from pkg.facenet import get_face_embedding
from pkg.utils import get_clusters

#######################################################################

def select_lfw_names(n_familiars, n_strangers, range_familiars, range_strangers, seed):
    """ Returns a list of names that constitutes the album and a list of names
    that are not in the album. The number of out-of-album names is equal to
    the number of in-album names with more than one image available.
    
    Arguments:
    - n_familiars (int): no. of persons with many images available
    - n_strangers (int): no. of persons with less than 6 pics availabale
    """
    fpath = os.path.join(os.path.dirname(__file__),
                         '..', '..', 'data', 'lfw', 'lfw-names.txt')
    names = pd.read_table(fpath, sep='\t', header=0, names=['name', 'count'])
    # select familiars to appear in album
    familiars_filter = (names['count'] >= range_familiars[0]) \
        & (names['count'] <= range_familiars[1])
    familiars = names[familiars_filter].sample(n=n_familiars, random_state=seed)
    # select strangers to appear in album
    strangers_filter =  (names['count'] >= range_strangers[0]) \
        & (names['count'] <= range_strangers[1])
    strangers = names[strangers_filter].sample(n=n_strangers, random_state=seed)
    # create album names
    in_album_names = list(familiars['name']) + list(strangers['name'])
    # no. of names in album with more than one image
    n_in_album = n_familiars + sum(strangers['count'] > 1)
    # list of names out of album for face recognition test set
    out_of_album_filter = ~names['name'].isin(in_album_names) 
    out_of_album = names[out_of_album_filter].sample(n=n_in_album, random_state=seed)
    out_of_album_names = list(out_of_album['name'])

    return in_album_names, out_of_album_names


def extract_lfw_faces(names, extract_mode=None):
    """ Extract faces from list of names.

    Arguments:
    - names (list)
    - extract_mode:
        - if 'test_out' extract exactly one image for each name
        - if 'test_in' extract last image for each name if more than one image, else none
        - else extract all images for each name except the last one if more than one image
    
    Output:
    - pytorch Tensor of size (n_samples, 3, 160, 160)
    - list of labels of size n_samples with labels in [0, len(names))
        or all equal to -1 if extract_mode='test_out'
    """
    faces = []
    labels = []
    lfw_path = os.path.join( os.path.dirname(__file__), '..', '..', 'data', 'lfw')
    mtcnn = MTCNN(device='cpu').eval()
    # create list of images and labels
    for name in names:
        # create list of image paths for current name
        fpaths = [os.path.join(lfw_path, name, f)
                     for f in os.listdir(os.path.join(lfw_path, name))
                     if os.path.isfile(os.path.join(lfw_path, name, f))]
        # select images to extract according to extract_mode and no. of available pics
        n_samples = len(fpaths)
        if extract_mode == 'test_out':
            fpaths = fpaths[-1:]
        elif extract_mode == 'test_in':
            if n_samples > 50:
                fpaths = fpaths[-10:]
            elif n_samples > 1:
                fpaths = fpaths[-1:]
            else:
                fpaths = []
        else:
            if n_samples > 50:
                fpaths = fpaths[:-10]
            elif n_samples > 1:
                fpaths = fpaths[:-1]
        # define label 
        if extract_mode == 'test_out':
            label = -1
        else:
            label = max(labels, default=-1) + 1
        # open and extract images
        for fpath in fpaths:
            img = Image.open(fpath)
            with torch.no_grad():
                face, prob = mtcnn(img, return_prob=True)
            if face is not None and prob > 0.99:
                faces.append(face)
                labels.append(label)

    return torch.stack(faces), labels


def load_lfw_embeddings(n_familiars, n_strangers, range_familiars, range_strangers,
                        seed, return_all=False):
    names_in, names_out = select_lfw_names(n_familiars, n_strangers,
        range_familiars, range_strangers, seed)

    print("Extracting faces")
    t0 = time()
    faces, labels_true = extract_lfw_faces(names_in)
    clusters_true = get_clusters(labels_true)
    print("Done ({:.2f}s)".format(time() - t0))
    n_clusters = len(clusters_true)
    print("No. of persons in album: {}".format(n_clusters))
    print("No. of faces extracted from album: {}".format(len(faces)))

    print()
    print("Computing face embeddings")
    t0 = time()
    embeddings = get_face_embedding(faces, batch_size=0.25)
    print("Done ({:.2f}s)".format(time() - t0))
    print("Embedding dimension: {}".format(embeddings.shape[1]))

    if return_all:
        return embeddings, clusters_true, names_in, names_out
    else:
        return embeddings, clusters_true

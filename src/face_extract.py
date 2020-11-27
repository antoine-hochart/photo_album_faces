import os
import numpy as np
import torch

from time import time
from tqdm import tqdm
from PIL import Image

from pkg.mtcnn import MTCNN, extract_face

######################################################################

ALBUM_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'album')
FACES_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'faces')

######################################################################

def extract_album_faces():
    """ Extract all the faces from the album that have a detection probability > 99%
    and save them in a `data/faces` folder.
    """
    np.random.seed(0)
    os.makedirs(os.path.abspath(FACES_PATH), exist_ok=True)
    mtcnn = MTCNN(select_largest=False, keep_all=True, device='cpu').eval()
    for root, dirs, files in os.walk(ALBUM_PATH):
        for fname in tqdm(files, ascii=True):
            fpath = os.path.join(ALBUM_PATH, fname)
            img = Image.open(fpath)
            with torch.no_grad():
                boxes, probs = mtcnn.detect(img, landmarks=False)
            if boxes is not None:
                for box, prob in zip(boxes, probs):
                    if prob > 0.99:
                        isfile = True
                        while isfile: 
                            rand_key = np.random.randint(10**5, 10**6)
                            save_path = os.path.join(FACES_PATH, '{}.png'.format(rand_key))
                            isfile = os.path.isfile(save_path)
                        _ = extract_face(img, box, save_path=save_path)


######################################################################

if __name__ == "__main__":
    print("Extracting faces from photo album")
    t0 = time()
    extract_album_faces()
    print("Done ({:.2f}s)".format(time() - t0))

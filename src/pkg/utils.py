import os
import shutil
import tempfile
import numpy as np

from urllib.request import urlopen, Request
from tqdm.auto import tqdm

#######################################################################

def download_url_to_file(url, dst, progress=True):
    """ Download object at the given URL to a local path.
    
    Arguments
    - url (string): URL of the object to download
    - dst (string): Full path where object will be saved, e.g. `/tmp/temporary_file`
    - progress (bool, optional): whether or not to display a progress bar to stderr
        Default: True
    """
    file_size = None
    # We use a different API for python2 since urllib(2) doesn't recognize the CA
    # certificates in older Python
    req = Request(url, headers={"User-Agent": "torch.hub"})
    u = urlopen(req)
    meta = u.info()
    if hasattr(meta, 'getheaders'):
        content_length = meta.getheaders("Content-Length")
    else:
        content_length = meta.get_all("Content-Length")
    if content_length is not None and len(content_length) > 0:
        file_size = int(content_length[0])

    # We deliberately save it in a temp file and move it after
    # download is complete. This prevents a local working checkpoint
    # being overridden by a broken download.
    dst = os.path.expanduser(dst)
    dst_dir = os.path.dirname(dst)
    f = tempfile.NamedTemporaryFile(delete=False, dir=dst_dir)

    try:
        with tqdm(total=file_size, disable=not progress, ascii=True,
                  unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            while True:
                buffer = u.read(8192)
                if len(buffer) == 0:
                    break
                f.write(buffer)
                pbar.update(len(buffer))
        f.close()
        shutil.move(f.name, dst)
    finally:
        f.close()
        if os.path.exists(f.name):
            os.remove(f.name)

#######################################################################

def get_clusters(labels):
    clusters = [[] for _ in np.unique(labels)]
    for i, label in enumerate(labels):
        clusters[int(label)].append(i) 
    return clusters

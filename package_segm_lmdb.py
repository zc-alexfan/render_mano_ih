from tqdm import tqdm
import numpy as np
import lmdb
import os.path as op
import cv2 as cv
import pyarrow as pa
from glob import glob


def dumps_pyarrow(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def fetch_lmdb_reader(db_path):
    env = lmdb.open(
        db_path,
        subdir=op.isdir(db_path),
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    txn = env.begin(write=False)
    return txn


def read_lmdb_image(txn, fname):
    image_bin = txn.get(fname.encode("ascii"))
    if image_bin is None:
        return image_bin
    image = np.fromstring(image_bin, dtype=np.uint8)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    return image


def package_lmdb(lmdb_name, map_size, fnames, keys, write_frequency=5000):
    """
    Package image files into a lmdb database.
    fnames are the paths to each file and also the key to fetch the images.
    lmdb_name is the name of the lmdb database file
    map_size: recommended to set to len(fnames)*num_types_per_image*10
    keys: the key of each image in dict
    """
    assert len(fnames) == len(keys)
    db = lmdb.open(lmdb_name, map_size=map_size)
    txn = db.begin(write=True)
    for idx, (fname, key) in tqdm(enumerate(zip(fnames, keys)), total=len(fnames)):
        img = cv.imread(fname)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        status, encoded_image = cv.imencode(".png", img, [cv.IMWRITE_JPEG_QUALITY, 100])
        assert status
        txn.put(key.encode("ascii"), encoded_image.tostring())

        if idx % write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)

    txn.commit()
    with db.begin(write=True) as txn:
        txn.put(b"__keys__", dumps_pyarrow(fnames))
        txn.put(b"__len__", dumps_pyarrow(len(fnames)))
    db.sync()
    db.close()
    print("Saved LMDB to " + lmdb_name)


if __name__ == "__main__":

    DB_NAME = "segm_32"
    db_path = DB_NAME + ".lmdb"
    fnames = glob("./outputs/segms/*/*/*/*/*")

    map_size = len(fnames) * 5130240
    keys = [fname.replace("./outputs/segms/", "") for fname in fnames]
    package_lmdb(db_path, map_size, fnames, keys)

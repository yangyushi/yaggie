import pickle
import os

def load_large_pkl(fn):
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(fn)
    with open(fn, 'rb') as f:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f.read(max_bytes)
    return bytes_in 

def read_pkl(fn):
    try:
        with open(fn, 'rb') as f:
            array = pickle.load(f)
        return array
    except OSError:
        print('lading large pkl...')
        bytes_in = load_large_pkl(fn)
        array = pickle.loads(bytes_in)
        return array

def read_meta(fn):
    with open(fn, 'rb') as f:
        metadata = pickle.load(f)
    return metadata

def read_xml(fn):
    with open(fn, 'r') as f:
        xml = f.read()
    return xml

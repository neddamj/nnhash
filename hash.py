'''
    Usage: python3 hashing.py -i path/to/image
'''

import subprocess
import argparse

def compute_hash(image_path):
    hashing_file_name = './nehcalc'
    output = subprocess.check_output([hashing_file_name, image_path])
    return output.strip().split()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
        help="path to input image to be hashed")
    args = vars(ap.parse_args())

    path = args['input']
    hash = compute_hash(path)
    print(hash)
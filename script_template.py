import gzip
import base64
import os
from pathlib import Path
from typing import Dict


# this is base64 encoded source code
file_data: Dict = {file_data}


for path, encoded in file_data.items():
    print(path)
    path = Path(path)
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(gzip.decompress(base64.b64decode(encoded)))


def run(command):
    os.system('echo "from setuptools import setup; setup(name=\'src\', packages=[\'src\'],)" > setup.py')
    os.system('export PYTHONPATH=${PYTHONPATH}:/kaggle/working && ' + command)


run('python setup.py develop --install-dir /kaggle/working')
run('python src/main.py')

from src import utils

FEATURES_DIR = "features"

utils.Feature.dir = FEATURES_DIR
if not os.path.exists(FEATURES_DIR):
    os.mkdir(FEATURES_DIR)
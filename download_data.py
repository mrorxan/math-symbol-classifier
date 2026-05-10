import os
import urllib.request
import tarfile

DATA_DIR = 'data'
URL = 'https://zenodo.org/records/259444/files/HASYv2.tar.bz2?download=1'
ARCHIVE = os.path.join(DATA_DIR, 'HASYv2.tar.bz2')

os.makedirs(DATA_DIR, exist_ok=True)

if os.path.exists(os.path.join(DATA_DIR, 'hasy-data')):
    print('Dataset already extracted, skipping.')
else:
    if not os.path.exists(ARCHIVE):
        print('Downloading HASYv2 (~33MB)...')
        urllib.request.urlretrieve(URL, ARCHIVE)
        print('  done.')
    else:
        print('Archive already downloaded.')

    print('Extracting...')
    with tarfile.open(ARCHIVE, 'r:bz2') as tar:
        tar.extractall(DATA_DIR)
    print('  done.')

print(f'Dataset ready in {DATA_DIR}/')

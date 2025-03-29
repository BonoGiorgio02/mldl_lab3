import os
import urllib.request
import zipfile
import shutil

# URL del dataset
dataset_url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
dataset_zip = "tiny-imagenet-200.zip"
dataset_folder = "dataset/"

# Creiamo la cartella per il dataset (se non esiste)
os.makedirs(dataset_folder, exist_ok=True)

# Scarichiamo il dataset (se non è già presente)
if not os.path.exists(dataset_zip):
    print("Downloading Tiny ImageNet dataset...")
    urllib.request.urlretrieve(dataset_url, dataset_zip)
    print("Download complete.")
else:
    print("Dataset already downloaded.")

# Estraiamo il file ZIP nella cartella dataset/
with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
    zip_ref.extractall(dataset_folder)
    print("Extraction complete.")

# Opzionale: eliminare il file ZIP per risparmiare spazio
os.remove(dataset_zip)
print("ZIP file deleted.")

# -----------------------------------------------------------------------------
# Adjust the format of the val split of the dataset to be used with ImageFolder
# -----------------------------------------------------------------------------
val_dir = "dataset/tiny-imagenet-200/val"
annotations_file = os.path.join(val_dir, "val_annotations.txt")

# Leggiamo il file delle annotazioni e riorganizziamo le immagini in cartelle per classe
with open(annotations_file, "r") as f:
    for line in f:
        fn, cls, *_ = line.split('\t')
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
        shutil.move(os.path.join(val_dir, "images", fn), os.path.join(val_dir, cls, fn))

# Rimuoviamo la cartella images vuota
shutil.rmtree(os.path.join(val_dir, "images"))

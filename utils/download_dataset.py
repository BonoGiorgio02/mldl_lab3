import os
import urllib.request
import zipfile
import shutil
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

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

transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    # mean and standard deviation for this dataset
])

# root folder / class name / name of the file of the parameters
# root/{classX}/x001.jpg
#root is the folder that contains all the data for the model
tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)

print(f"Length of train dataset: {len(tiny_imagenet_dataset_train)}")
print(f"Length of val dataset: {len(tiny_imagenet_dataset_val)}")

# During training we are introducing some noise
train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True, num_workers=8)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)
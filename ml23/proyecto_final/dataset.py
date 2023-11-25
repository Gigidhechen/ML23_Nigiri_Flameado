import pathlib
from typing import Any, Callable, Optional, Tuple
import torch
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import pandas as pd
import cv2
import os
import numpy as np
from utils import to_numpy, to_torch, add_img_text, get_transforms
import json
import matplotlib.pyplot as plt


file_path = pathlib.Path(__file__).parent.absolute()

def get_loader(split, batch_size, shuffle=True, num_workers=0):
    '''
    Get train and validation loaders
    args:
        - batch_size (int): batch size
        - split (str): split to load (train, test or val)
    '''
    if(split=="train"):
        split=True
    dataset = datasets.MNIST(root=file_path,
                      train=split,
                      download=True,
                      transform=transforms.ToTensor())
    dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
        )
    return dataset, dataloader

def main():
    # Visualizar de una en una imagen
    split = "train"
    dataset, dataloader = get_loader(split=split, batch_size=1, shuffle=False)
    print(f"Loading {split} set with {len(dataloader)} samples")
    for datapoint in dataloader:
        #transformed = datapoint['transformed']
        original = datapoint[0]
        label = datapoint[1]
        

        # Si se aplico alguna normalizacion, deshacerla para visualizacion

        # Transformar a numpy
        original = to_numpy(original)  # 0 - 255
        # transformed = (transformed * 255).astype('uint8')  # 0 - 255

        # Aumentar el tamaño de la imagen para visualizarla mejor
        viz_size = (200, 200)
        original = cv2.resize(original, viz_size)

        # Concatenar las imagenes, tienen que ser del mismo tipo
        original = original.astype('float32') / 255
        np_img = np.expand_dims(original,-1)

        np_img = add_img_text(np_img, str(label.item()))

        cv2.imshow("algo",np_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
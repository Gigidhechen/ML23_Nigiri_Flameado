import pathlib
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision import transforms
import cv2
import numpy as np
from utils import to_numpy, to_torch, add_img_text, get_transforms
import matplotlib.pyplot as plt


file_path = pathlib.Path(__file__).parent.absolute()

def get_loader(split, batch_size, shuffle=True, num_workers=0):
    '''
    Get train and validation loaders
    args:
        - batch_size (int): batch size
        - split (str): split to load (train, test or val)
    '''
    _trainign = split == "train"
    dataset = datasets.MNIST(root=file_path,
                      train=_trainign,
                      download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          transforms.RandomInvert(),
                          transforms.GaussianBlur(kernel_size=(3,5),sigma=(0.1,0.2)),
                          transforms.RandomRotation(degrees=(0,25)),
                          transforms.ColorJitter(brightness=0.5, hue=0.3)
                          ]))
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
        original = datapoint[0]
        label = datapoint[1]
        


        original = to_numpy(original) 
        
        # Aumentar el tamaño de la imagen para visualizarla mejor
        viz_size = (300, 300)
        original = cv2.resize(original, viz_size)

        original = original.astype('float32')
        np_img = np.expand_dims(original,-1)

        np_img = add_img_text(np_img, str(label.item()))

        cv2.imshow("algo",np_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
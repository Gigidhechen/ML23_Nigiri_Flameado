import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from network import Network
import torch
from utils import to_numpy, get_transforms, add_img_text
from dataset import EMOTIONS_MAP
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()

def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path)
    val_transforms, unnormalize = get_transforms("test", img_size = 48)
    tensor_img = val_transforms(img)
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized

def predict(img_title_paths):
    '''
        Hace la inferencia de las imagenes
        args:
        - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    '''
    # Cargar el modelo
    modelo = Network(48, 7)
    modelo.load_model("modelo16Train.pth")
    for path in img_title_paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)

        # Inferencia
        logits, proba = modelo.predict(transformed)
        pred = torch.argmax(proba, -1).item()
        pred_label = EMOTIONS_MAP[pred]

        # Original / transformada
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        img = add_img_text(img, f"Pred: {pred_label}")

        # Mostrar la imagen
        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.imshow("Predicción - transformed", denormalized)
        cv2.waitKey(0)

if __name__=="__main__":
    # Direcciones relativas a este archivo
    img_paths = [
                 "./test_imgs/Brandon.jpg",
                 "./test_imgs/Disgusto 1.png",
                 "./test_imgs/Disgusto 2.png",
                 "./test_imgs/Disgusto 3.png",
                 "./test_imgs/Disgusto 4.png",
                 "./test_imgs/Enojo 1.png",
                 "./test_imgs/Enojo 2.png",
                 "./test_imgs/Feliz 1.png",
                 "./test_imgs/Feliz 2.png",
                 "./test_imgs/Feliz 3.png",
                 "./test_imgs/Feliz 4.png",
                 "./test_imgs/Feliz 5.png",
                 "./test_imgs/Miedo 1.png",
                 "./test_imgs/Neutro 1.png",
                 "./test_imgs/Neutro 2.png",
                 "./test_imgs/Neutro 3.png",
                 "./test_imgs/Neutro 4.png",
                 "./test_imgs/Neutro 5.png",
                 "./test_imgs/Sorpresa 1.png",
                 "./test_imgs/Sorpresa 2.png",
                 "./test_imgs/Sorpresa 3.png",
                 "./test_imgs/Sorpresa 4.png",
                 "./test_imgs/Sorpresa 5.png",
                 "./test_imgs/Sorpresa Juan.png",]
    predict(img_paths)
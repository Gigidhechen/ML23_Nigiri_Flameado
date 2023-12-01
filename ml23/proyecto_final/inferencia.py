import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from network import Network
import torch
from utils import to_numpy, get_transforms, add_img_text
import pathlib

file_path = pathlib.Path(__file__).parent.absolute()
def read_transparent_img(img):
    image_4channel = img
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.float32)


def load_img(path):
    assert os.path.isfile(path), f"El archivo {path} no existe"
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img.shape[-1] == 4:
        img = read_transparent_img(img)
    img = img /255

    val_transforms, unnormalize = get_transforms("test", img_size = 28)
    tensor_img = val_transforms(img.astype('float32'))
    denormalized = unnormalize(tensor_img)
    return img, tensor_img, denormalized

def predict(img_title_paths):
    '''
        Hace la inferencia de las imagenes
        args:
        - img_title_paths (dict): diccionario con el titulo de la imagen (key) y el path (value)
    '''
    # Cargar el modelo
    modelo = Network(28, 10)
    modelo.load_model("modelo2.pth")
    for path in img_title_paths:
        # Cargar la imagen
        # np.ndarray, torch.Tensor
        im_file = (file_path / path).as_posix()
        original, transformed, denormalized = load_img(im_file)

        # Inferencia
        logits, proba = modelo.predict(transformed)
        pred = torch.argmax(proba, -1).item()
        pred_label = str(pred)

        # Original / transformada
        h, w = original.shape[:2]
        resize_value = 300
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        if len(img.shape) < 3:
            img=np.expand_dims(img,-1)
        img = add_img_text(img * 255, f"Pred: {pred_label}")

        # Mostrar la imagen
        denormalized = to_numpy(denormalized)
        denormalized = cv2.resize(denormalized, (resize_value, resize_value))
        cv2.imshow("Predicción - original", img)
        cv2.waitKey(0)

if __name__=="__main__":
    # Direcciones relativas a este archivo
    img_paths = [
                 "./test_imgs/1.jpg",
                 "./test_imgs/3.png"
                 ]
    predict(img_paths)
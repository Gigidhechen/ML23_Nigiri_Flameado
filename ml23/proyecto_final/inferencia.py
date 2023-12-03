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
    img=img.astype(('float32'))
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
    modelo.load_model("modelo1.pth")
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
        resize_value = 500
        img = cv2.resize(original, (w * resize_value // h, resize_value))
        if len(img.shape) < 3:
            img=np.expand_dims(img,-1)
        img = add_img_text(img * 255, f"Pred: {pred_label}")

        # Mostrar la imagen
        cv2.imshow("Predicción - original", img)
        cv2.waitKey(0)
                                        
if __name__=="__main__":
    # Direcciones relativas a este archivo
    img_paths = [
                 "./test_imgs/0.jpg",
                 "./test_imgs/1.jpg",
                 "./test_imgs/2.jpg",
                 "./test_imgs/3.jpg",
                 "./test_imgs/4.jpg",
                 "./test_imgs/5.jpg",
                 "./test_imgs/6.jpg",
                 "./test_imgs/7.jpg",
                 "./test_imgs/8.jpg",
                 "./test_imgs/9.jpg",
                 "./test_imgs/Azul_0.jpg",
                 "./test_imgs/Azul_1.jpg",
                 "./test_imgs/Azul_2.jpg",
                 "./test_imgs/Azul_3.jpg",
                 "./test_imgs/Azul_4.jpg",
                 "./test_imgs/Azul_5.jpg",
                 "./test_imgs/Azul_6.jpg",
                 "./test_imgs/Azul_7.jpg",
                 "./test_imgs/Azul_8.jpg",
                 "./test_imgs/Azul_9.jpg",
                 "./test_imgs/Blur_0.jpg",
                 "./test_imgs/Blur_1.jpg",
                 "./test_imgs/Blur_2.jpg",
                 "./test_imgs/Blur_3.jpg",
                 "./test_imgs/Blur_4.jpg",
                 "./test_imgs/Blur_5.jpg",
                 "./test_imgs/Blur_6.jpg",
                 "./test_imgs/Blur_7.jpg",
                 "./test_imgs/Blur_8.jpg",
                 "./test_imgs/Blur_9.jpg",
                 "./test_imgs/Noise_0.jpg",
                 "./test_imgs/Noise_1.jpg",
                 "./test_imgs/Noise_2.jpg",
                 "./test_imgs/Noise_3.jpg",
                 "./test_imgs/Noise_4.jpg",
                 "./test_imgs/Noise_5.jpg",
                 "./test_imgs/Noise_6.jpg",
                 "./test_imgs/Noise_7.jpg",
                 "./test_imgs/Noise_8.jpg",
                 "./test_imgs/Noise_9.jpg",
                 "./test_imgs/Pixel_0.jpg",
                 "./test_imgs/Pixel_1.jpg",
                 "./test_imgs/Pixel_2.jpg",
                 "./test_imgs/Pixel_3.jpg",
                 "./test_imgs/Pixel_4.jpg",
                 "./test_imgs/Pixel_5.jpg",
                 "./test_imgs/Pixel_6.jpg",
                 "./test_imgs/Pixel_7.jpg",
                 "./test_imgs/Pixel_8.jpg",
                 "./test_imgs/Pixel_9.jpg",
                 "./test_imgs/Rojo_0.jpg",
                 "./test_imgs/Rojo_1.jpg",
                 "./test_imgs/Rojo_2.jpg",
                 "./test_imgs/Rojo_3.jpg",
                 "./test_imgs/Rojo_4.jpg",
                 "./test_imgs/Rojo_5.jpg",
                 "./test_imgs/Rojo_6.jpg",
                 "./test_imgs/Rojo_7.jpg",
                 "./test_imgs/Rojo_8.jpg",
                 "./test_imgs/Rojo_9.jpg",
                 "./test_imgs/Rotate_0.jpg",
                 "./test_imgs/Rotate_1.jpg",
                 "./test_imgs/Rotate_2.jpg",
                 "./test_imgs/Rotate_3.jpg",
                 "./test_imgs/Rotate_4.jpg",
                 "./test_imgs/Rotate_5.jpg",
                 "./test_imgs/Rotate_6.jpg",
                 "./test_imgs/Rotate_7.jpg",
                 "./test_imgs/Rotate_8.jpg",
                 "./test_imgs/Rotate_9.jpg",
                 "./test_imgs/Verde_0.jpg",
                 "./test_imgs/Verde_1.jpg",
                 "./test_imgs/Verde_2.jpg",
                 "./test_imgs/Verde_3.jpg",
                 "./test_imgs/Verde_4.jpg",
                 "./test_imgs/Verde_5.jpg",
                 "./test_imgs/Verde_6.jpg",
                 "./test_imgs/Verde_7.jpg",
                 "./test_imgs/Verde_8.jpg",
                 "./test_imgs/Verde_9.jpg",
                 ]
    predict(img_paths)
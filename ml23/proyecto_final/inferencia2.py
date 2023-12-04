import cv2
import matplotlib.pyplot as plt
from network import Network
import os
import pathlib
import numpy as np
import torch
import torch.nn as nn
import torchvision
from dataset import get_loader
from utils import to_numpy, get_transforms, add_img_text


file_path = pathlib.Path(__file__).parent.absolute()
def predict(img_title_paths):
    modelo = Network(28, 10)
    modelo.load_model("modelo3.pth")
    for path in img_title_paths:
        im_file = (file_path / path).as_posix()
        assert os.path.isfile(im_file), f"El archivo {path} no existe"
        test_img=cv2.imread(im_file,cv2.IMREAD_GRAYSCALE)

        normalizedImg=np.zeros((800,800))
        normalizedImg=cv2.normalize(test_img,normalizedImg,0,255,cv2.NORM_MINMAX)  
        img = normalizedImg /255
        h, w = img.shape[:2]
        resize_value = 300
        img = cv2.resize(img, (w * resize_value // h, resize_value))
        img_resized = cv2.resize(test_img,(28,28),interpolation=cv2.INTER_LINEAR)
        img_resized=cv2.bitwise_not(img_resized)


        img_resized=img_resized.astype('float32')
        transformation=torchvision.transforms.ToTensor()
        img_resized=(transformation(img_resized))
        logits, proba = modelo.predict(img_resized)
        pred = torch.argmax(proba, -1).item()
        pred_label = str(pred)
        if len(img.shape) < 3:
            img=np.expand_dims(img,-1)
        img = add_img_text(img.astype('float32'), f"Pred: {pred_label}")

        cv2.imshow("prueba",img)
        cv2.waitKey(0)



if __name__=="__main__":
    # Direcciones relativas a este archivo
    modelo = Network(28, 10)
    modelo.load_model("modelo4.pth")
    cost_function=nn.CrossEntropyLoss()
    val_dataset, val_loader = \
        get_loader("val",
                    batch_size=1,
                    shuffle=False)
    for i, batch in enumerate(val_loader, 0):
        batch_imgs = batch[0]
        batch_labels = batch[1].cuda()
        device = modelo.device
        batch_labels = batch_labels.to(device)
        with torch.inference_mode():
            # TODO: realiza un forward pass, calcula el loss y acumula el costo
            outputs,proba=modelo(batch_imgs)
            costo= cost_function(outputs,batch_labels)
            pred = torch.argmax(proba, -1).item()
            pred_label = str(pred)

        img=batch_imgs.cpu() 
        img=img.numpy()
        img = np.transpose(img[0], (1, 2, 0))
        img = cv2.resize(img, (300, 300))
        img = add_img_text(img.astype('float32'), f"Pred: {pred_label}")
        # cv2.imshow("prueba",img)
        # cv2.waitKey(0)
    acc=sum(pred==batch_labels)*100
    
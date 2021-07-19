import warnings
warnings.simplefilter('ignore')


from keras.preprocessing import image
import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import os


from PIL import Image

import cv2

from torch.utils.data import Dataset




class ArtPaintDataset(Dataset):
    def __init__(self, df, transform=None):
          #  torch의 dataset을 상속 받기 때문에 Dataset 클래스 override 되지 않도록 init
          super().__init__()




if __name__ == '__main__':



    batch_size = 32
    img_height = 100
    img_width = 180




    # data = plt.imread('\\train\\elephant\\pic_020.jpg')

    image = Image.open("train\\elephant\\pic_020.jpg")

    #data = image.load_img('/train/elephant/pic_020.jpg',  target_size=(227,227))


    image.show()

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        'train\\elephant\\',
        validation_split=0.2
        ,subset="training"
        ,seed=123
        ,image_size=(227,227)
        ,batch_size=3
    )



    # plt.imshow(image)


import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
import itertools as it
import json
import codecs
from numpy import genfromtxt
from datetime import datetime
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import os
import timeit
import csv
from PIL import Image
from colorama import Fore, init

init(autoreset=True)

Image.MAX_IMAGE_PIXELS = 933120000

shape_x = 224
shape_y = 224

model = VGG16(weights='imagenet', include_top=False, input_shape=(shape_x, shape_y, 3))
# model.summary()

# images_list = os.listdir('/home/edoardo/prova/SuperRare/Images/')
images_list = os.listdir('/home/edoardo/prova/non si sa mai/')

token_list = []
count = 0


def load_image(path):
    global count
    count = count + 1
    print(Fore.RED + str(count))
    print(path)
    token = path
    tok = token.split("_")[0]
    inp = image.load_img('/home/edoardo/prova/SuperRare/Images/' + path, target_size=(shape_x, shape_y))
    token_list.append(tok)
    return image.img_to_array(inp)


image_arrays = []

for i in images_list:
    image_arrays.append(load_image(i))

df_artwork = pd.DataFrame.from_dict({"image_array": image_arrays})


def apply_pretrained(x, model1=model):
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = model1.predict(x)
    return features.flatten()


X = df_artwork[pd.notna(df_artwork.image_array)]['image_array'].apply(apply_pretrained)
Y = np.matrix(X.to_list())
print(Y.shape)
print(Y)

l2 = list(range(1, Y.shape[0] + 1))
pearson = np.corrcoef(Y)
ju = np.insert(pearson, 0, l2, axis=0)
print(ju)

np.savetxt('/media/edoardo/76DAD24FDAD20AEF/Users/edoar/Desktop/Edo/Uniud/2) Magistrale/Tesi/Fase '
           '2/Dati/immagini.csv', ju, delimiter=',')
with open('/media/edoardo/76DAD24FDAD20AEF/Users/edoar/Desktop/Edo/Uniud/2) Magistrale/Tesi/Fase '
          '2/Dati/token1.csv', 'w') as f:
    write = csv.writer(f)
    write.writerow(token_list)

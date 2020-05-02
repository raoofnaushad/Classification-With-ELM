from keras.datasets import mnist
from keras.utils import to_categorical
from elm import ELM, load_model
import argparse
import os
import numpy as np
from PIL import Image 
import numpy as np
import cv2
from tqdm import tqdm


def Dataset_loader(DIR, RESIZE, sigmaX=10):
    IMG = []
    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))
    for IMAGE_NAME in tqdm(os.listdir(DIR)):
        PATH = os.path.join(DIR,IMAGE_NAME)
        _, ftype = os.path.splitext(PATH)
        # if ftype == ".png":
        img = read(PATH)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
           
        img = cv2.resize(img, (RESIZE,RESIZE))
           
        IMG.append(np.array(img))
    return IMG


def softmax(x):
    c = np.max(x, axis=-1)
    upper = np.exp(x - c)
    lower = np.sum(upper, axis=-1)
    return upper / lower

test = np.array(Dataset_loader('data/check',28))
s = np.arange(test.shape[0])
np.random.shuffle(s)
test = test[s]

test = test.astype(np.float32) / 255.
test = test.reshape(-1, 28**2)


model = load_model('model.h5')

y = softmax(model.predict(test))
class_pred = np.argmax(y)
prob = y

if class_pred == 1:
    print("Beningant")
else:
    print("Malignant")

### (10000, 784)
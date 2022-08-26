import glob
import os
import cv2
import numpy as np
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

CATEGORIES = ["Axial", "Coronal", "Sagittal"]
IMG_SIZE = 256

files = glob.glob("D:/AxisDetection/ValidationData/*.png")

data_array = []

for dataFile in files:
    image = cv2.imread(dataFile, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    img = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    data_array.append(img)

model = tf.keras.models.load_model("D:/AxisDetection/Model/plane_model.h5")

length = len(data_array)

for i in range(length):
    prediction = model.predict([data_array[i]])
    array_x = np.asarray(prediction)
    index = 0
    for x in range(3):
        if int(array_x[0][x]) == 1:
            index = x
    print(CATEGORIES[index])

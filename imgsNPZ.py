import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import glob

imgDim = 256
colorMode = "grayscale"
colorChannels = 1

#imgDir = '/images_cropped/**/*.jpg'
imgDir = 'images cropped/**/*.png'
outputFile = 'ASD.npz'

imgCount = 0
for file in glob.iglob(imgDir, recursive=True):
    imgCount = imgCount + 1
print("Number of images:", imgCount)

X = np.zeros((imgCount, imgDim, imgDim, colorChannels), dtype=np.uint8)
y = np.zeros((imgCount, 1), dtype=np.uint8)

for i, imgFile in enumerate(glob.iglob(imgDir, recursive=True)):
    #print(imgFile)
    try:
        img = load_img(imgFile, color_mode=colorMode, target_size=(imgDim, imgDim))
    except Exception:
        pass
    img = img_to_array(img)
    X[i] = img

    label = None
    if "neg" in imgFile:
        label = 0
    elif "pos" in imgFile:
        label = 1

    y[i] = label

print("X shape:", X.shape)
print("y shape:", y.shape)

X = (X / 255.0) #Normalization

#Saving all into compressed Numpy array
np.savez_compressed(outputFile, X=X, y=y)
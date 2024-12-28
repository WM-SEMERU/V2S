import argparse
import os
import numpy as np

from PIL import Image
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--model_path")
parser.add_argument("--detection_path")
parser.add_argument("--frames")
args = parser.parse_args()

PATH_MODEL = args.model_path
PATH_DETECTION = args.detection_path

# This is the size defined in the model for AlexNet architecture
SIZE = 227

start_time = datetime.datetime.now().replace(microsecond=0)

model = load_model(args.model_path)
# List separated by comas
list_images = args.frames.split(',')
predictions = []
images = []

for i in list_images:
    touch = os.path.join(PATH_DETECTION, i)
    temp_image = load_img(touch)
    temp_image = temp_image.resize((SIZE, SIZE), Image.ANTIALIAS)
    temp_image_np = img_to_array(temp_image)
    # Normalization
    temp_image_np = temp_image_np.reshape((1,) + temp_image_np.shape) / 255
    images.append(temp_image_np)

images = np.array(images).reshape(len(images), SIZE, SIZE, 3)
predictions = model.predict(images)
# predictions = list(map(int, predictions))
for j in predictions:
    print(j)

end_time = datetime.datetime.now().replace(microsecond=0)
print('detection_cnn.py took: ' + str(end_time - start_time) + ' to run.')

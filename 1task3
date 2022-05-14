# import of library
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions

# setting
model = ResNet50(weights='imagenet')
target_size = (224, 224)

# function to predict rilaible person
def predict(model, img, target_size, top_n=3):
  if img.size != target_size:
    img = img.resize(target_size)
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return decode_predictions(preds, top=top_n)[0]

# function to display result
def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')
  plt.show()
  print('---------------------')
  print('Type of image')
  print('---------------------')
  print(preds[0][1])
  print('---------------------')
  print('Probability')
  print('---------------------')
  print(preds[0][2])
  print('---------------------')	

# to control through console
if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--image_url", help="url to image")
  args = a.parse_args()
  
  if args.image is None and args.image_url is None:
    a.print_help()
    sys.exit(1)

  if args.image is not None:
    img = Image.open(args.image)
    preds = predict(model, img, target_size)
    plot_preds(img, preds)

  if args.image_url is not None:
    response = requests.get(args.image_url)
    img = Image.open(BytesIO(response.content))
    preds = predict(model, img, target_size)
    plot_preds(img, preds)

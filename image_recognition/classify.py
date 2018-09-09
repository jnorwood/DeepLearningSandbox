# from:
# https://deeplearningsandbox.com/how-to-build-an-image-recognition-system-using-keras-and-tensorflow-for-a-1000-everyday-object-559856e04699
# https://github.com/DeepLearningSandbox/DeepLearningSandbox
# also using keras-squeezenet originally from: https://github.com/wohlert/keras-squeezenet , checked out under
# image_recognition 
# description of available models:
# https://keras.io/applications/#xception
import sys
import argparse
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from keras.preprocessing import image
# I have tested that each of these models works for the single jpg image.
# I added the keras-squeezenet model which wasn't available in the keras.applications models
#from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
#from keras.applications.nasnet import NASNetMobile, preprocess_input, decode_predictions
#from keras.applications.mobilenetv2 import MobileNetV2, preprocess_input, decode_predictions
#from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
#from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
#from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
## demnsenet failed from keras.applications.densenet import DenseNet201, preprocess_input, decode_predictions
#from keras.applications.xception import  Xception, preprocess_input, decode_predictions
#from keras.applications.nasnet import NASNetLarge, preprocess_input, decode_predictions
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
import sys
sys.path.insert(0, '/home/jay/DeepLearningSandbox/image_recognition/keras-squeezenet')

from squeezenet import SqueezeNet

#model = ResNet50(weights='imagenet')
#model = VGG16(weights='imagenet')
#model = NASNetMobile(weights='imagenet')
#model = MobileNetV2(weights='imagenet')
#model = MobileNet(weights='imagenet')
#model = InceptionResNetV2(weights='imagenet')
#model = InceptionV3(weights='imagenet')
## densenet failed model = DenseNet201(weights='imagenet')
#model = Xception(weights='imagenet')
#model = NASNetLarge(weights='imagenet')
model = SqueezeNet(weights='imagenet')

# Note that you need to change the target size for the different network models
target_size = (224, 224)
#target_size = (299, 299) #for InceptionResNetV2,InceptionV3, Xception
#target_size = (331, 331) # for NASNetLarge

def predict(model, img, target_size, top_n=3):
  """Run model prediction on image
  Args:
    model: keras model
    img: PIL format image
    target_size: (w,h) tuple
    top_n: # of top predictions to return
  Returns:
    list of predicted labels and their probabilities
  """
  if img.size != target_size:
    img = img.resize(target_size)

  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  preds = model.predict(x)
  return decode_predictions(preds, top=top_n)[0]

def plot_preds(image, preds):
  """Displays image and the top-n predicted probabilities in a bar graph
  Args:
    image: PIL image
    preds: list of predicted labels and their probabilities
  """
  plt.imshow(image)
  plt.axis('off')

  plt.figure()
  order = list(reversed(range(len(preds))))
  bar_preds = [pr[2] for pr in preds]
  labels = (pr[1] for pr in preds)
  plt.barh(order, bar_preds, alpha=0.5)
  plt.yticks(order, labels)
  plt.xlabel('Probability')
  plt.xlim(0,1.01)
  plt.tight_layout()
  plt.show()

if __name__=="__main__":
  a = argparse.ArgumentParser()
  a.add_argument("--image", help="path to image")
  a.add_argument("--image_url", help="url to image")
  #args = a.parse_args()
  #args = a.parse_args(['--image','/home/jay/DeepLearningSandbox/image_recognition/images/African_Bush_Elephant.jpg'])
  args = a.parse_args(['--image_url','http://i.imgur.com/wpxMwsR.jpg'])

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
    pass


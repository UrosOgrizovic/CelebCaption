import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import pyplot
from numpy import asarray
from mtcnn.mtcnn import MTCNN


def load_images(path="data/img/images_ct"):
    w, h = 224, 224
    i = 0
    with open(path, 'rb') as file:
        d = np.fromfile(file, dtype=np.uint8, count=w * h).reshape(h, w)


# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


if __name__ == "__main__":
    model = VGGFace()
    # load_images()
    img = extract_face('data/benstiller.jpg')
    img = np.array(img).astype(np.float)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # preprocessing the input increases accuracy
    x = utils.preprocess_input(x, version=1)
    print(x.shape)
    predicted = model.predict(x)
    print(predicted.shape)
    res = utils.decode_predictions(predicted)
    print(res)

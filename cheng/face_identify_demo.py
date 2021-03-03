from keras.engine import Model
from keras import models
from keras import layers
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import numpy as np
from scipy import spatial
import cv2
import os
import glob
import pickle
from time import sleep
from precompute_features import FaceExtractor
import face_helpers


def load_stuff(filename):
    saved_stuff = open(filename, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


class FaceIdentify:
    """
    Singleton class for real time face identification
    """
    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"

    def __init__(self, precompute_features_file=None):
        self.face_size = 224
        self.precompute_features_map = load_stuff(precompute_features_file)
        self.model = VGGFace(model='resnet50',
                             include_top=False,
                             input_shape=(224, 224, 3),
                             pooling='avg')  # pooling: None, avg or max
        self.img_model = VGGFace()

    def draw_label(self, image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
        size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        x, y = point
        cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
        cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)

    def identify_face(self, features, threshold=100):
        """Identify face from features.

        Args:
            features (array_like): facial features
            threshold (int, optional): Distance threshold. Defaults to 100.

        Returns:
            string: The name of the person whose facial features are the closest
            to the facial features that were passed as an argument.
            If no face is close enough, returns "?".
        """
        distances = []
        for person in self.precompute_features_map:
            person_features = person.get("features")
            distance = spatial.distance.euclidean(person_features, features)
            distances.append(distance)
        min_distance_value = min(distances)
        min_distance_index = distances.index(min_distance_value)
        if min_distance_value < threshold:
            return self.precompute_features_map[min_distance_index].get("name")
        else:
            return "?"

    def detect_face(self):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture("..\\data\\test\\video\\Bradley Cooper, Lady Gaga\\Bradley Cooper, Lady Gaga.mp4")
        frame_index = -1
        x, y, w, h = 0, 0, 0, 0
        # infinite loop, break by key ESC
        while video_capture.isOpened():
            frame_index += 1
            # Capture frame-by-frame
            status, frame = video_capture.read()
            if not status:
                # video over
                break
            if frame_index % 10 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=10,
                    minSize=(64, 64)
                )
                # placeholder for cropped faces
                faces_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
                for i, face in enumerate(faces):
                    face_img, cropped = face_helpers.crop_face(frame, face, margin=10, size=self.face_size)
                    (x, y, w, h) = cropped
                    faces_imgs[i, :, :, :] = face_img
                if len(faces_imgs) > 0:
                    # generate features for each face
                    features_faces = self.model.predict(faces_imgs)
                    predicted_names = [self.identify_face(features_face) for features_face in features_faces]
                    # preprocessing the input increases accuracy
                    # faces_imgs = utils.preprocess_input(faces_imgs, version=1)
                    for i in range(len(predicted_names)):
                        if predicted_names[i] == "?":
                            features_face = self.img_model.predict(np.expand_dims(faces_imgs[i], axis=0))
                            res = utils.decode_predictions(features_face)
                            # take prediction with highest prob
                            predicted_names[i] = res[0][0][0][2:-1] # [2:-1] is to remove b' and '

            # draw rectangle around head
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
            # write names
            for i, face in enumerate(faces):
                label = "{}".format(predicted_names[i])
                self.draw_label(frame, (face[0], face[1]), label)

            cv2.imshow('Keras Faces', frame)
            if cv2.waitKey(5) == 27:  # ESC key press
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()



def main():
    face = FaceIdentify(precompute_features_file="./data/precompute_features.pickle")
    face.detect_face()

if __name__ == "__main__":
    main()
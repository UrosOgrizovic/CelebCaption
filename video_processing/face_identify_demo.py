import os
from keras.engine import Model
from keras import models
from keras import layers
from keras.layers import Input
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import numpy as np
from scipy import spatial
import cv2
import glob
import pickle
from time import sleep
import face_helpers
import video_constants as const


def load_stuff(filename):
    saved_stuff = open(filename, "rb")
    stuff = pickle.load(saved_stuff)
    saved_stuff.close()
    return stuff


class FaceIdentify:
    HAAR_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"

    def __init__(self, input_video_path, output_video_path, precompute_features_file=None):
        self.input_video_path = input_video_path
        self.output_video_path = output_video_path
        self.face_size = 224
        self.precompute_features_map = load_stuff(precompute_features_file)
        self.model = VGGFace(model='resnet50',
                             include_top=False,
                             input_shape=(224, 224, 3),
                             pooling='avg')  # pooling: None, avg or max
        self.img_model = VGGFace()

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
        face_cascade = cv2.CascadeClassifier(self.HAAR_PATH)

        # 0 means the default video capture device in OS
        video_capture = cv2.VideoCapture(self.input_video_path)
        frame_index = -1
        face_identification_period = 10  # face detection will be done every n frames

        fps = video_capture.get(cv2.CAP_PROP_FPS)
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_output = cv2.VideoWriter(self.output_video_path, fourcc, fps, (width, height))
        print('Processing video...')
        # infinite loop, break by key ESC
        while video_capture.isOpened():
            frame_index += 1
            # Capture frame-by-frame
            status, frame = video_capture.read()
            if not status:
                # video over
                break
            if frame_index % face_identification_period == 0:
                gray, faces = face_helpers.instantiate_gray_and_faces(frame, face_cascade)
                faces_imgs, rectangles_to_draw = self.extract_faces(frame, faces)
                if len(faces_imgs) > 0:
                    predicted_names = self.predict_names(faces_imgs)

            draw_rectangles(frame, rectangles_to_draw)

            # write names
            for i, face in enumerate(faces):
                draw_label(frame, (face[0], face[1]), predicted_names[i])

            video_output.write(frame)

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()
        print('Video processing completed successfully')

    def extract_faces(self, frame, faces):
        # placeholder for cropped faces
        faces_imgs = np.empty((len(faces), self.face_size, self.face_size, 3))
        rectangles_to_draw = []
        for i, face in enumerate(faces):
            face_img, cropped = face_helpers.crop_face(frame, face, margin=10, size=self.face_size)
            rectangles_to_draw.append(cropped)
            faces_imgs[i, :, :, :] = face_img
        return faces_imgs, rectangles_to_draw

    def predict_names(self, faces_imgs):
        # generate features for each face
        features_faces = self.model.predict(faces_imgs)
        predicted_names = [self.identify_face(features_face) for features_face in features_faces]

        for i in range(len(predicted_names)):
            if predicted_names[i] == "?":
                features_face = self.img_model.predict(np.expand_dims(faces_imgs[i], axis=0))
                res = utils.decode_predictions(features_face)
                # take prediction with highest prob
                predicted_names[i] = res[0][0][0][2:-1] # [2:-1] is to remove b' and ' from string
        return predicted_names

def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=1, thickness=2):
    size = cv2.getTextSize(label, font, font_scale, thickness)[0]
    x, y = point
    cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), const.LABEL_COLOR, cv2.FILLED)
    # cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), const.LABEL_COLOR, 2)
    cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)


def draw_rectangles(frame, rectangles_to_draw):
    for rect in rectangles_to_draw:
        (x, y, w, h) = rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), const.RECTANGLE_COLOR, 2)


if __name__ == "__main__":
    input_video_path = "..\\data\\test\\video\\cooper gaga.mp4"
    # input_video_path = "..\\data\\test\\video\\fallon stiller.mp4"
    output_video_path = "../processed_video.mp4"
    face = FaceIdentify(input_video_path, output_video_path, precompute_features_file="./data/precompute_features.pickle")
    face.detect_face()

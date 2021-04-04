import cv2
import os
import glob
import pickle
from keras_vggface.vggface import VGGFace
import numpy as np
from keras.preprocessing import image
from keras_vggface import utils
import face_helpers


def pickle_stuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


class FaceExtractor(object):
    """
    Singleton class to extract face images from video files
    """
    CASE_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"

    def __init__(self, face_size=224):
        self.face_size = face_size

    def extract_faces(self, video_file, save_folder):
        face_cascade = cv2.CascadeClassifier(self.CASE_PATH)

        # 0 means the default video capture device in OS
        cap = cv2.VideoCapture(video_file)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        print("length: {}, w x h: {} x {}, fps: {}".format(length, width, height, fps))
        face_detection_period = 10  # face detection will be done every n frames

        frame_index = -1
        # infinite loop, break by key ESC
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_index = frame_index + 1
                if frame_index % face_detection_period == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=1.2,
                        minNeighbors=10,
                        minSize=(64, 64)
                    )
                    # only keep the biggest face as the main subject
                    face = None
                    if len(faces) > 1:  # Get the largest face as main face
                        face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))  # area = w * h
                    elif len(faces) == 1:
                        face = faces[0]
                    if face is not None:
                        face_img, cropped = face_helpers.crop_face(frame, face, margin=40, size=self.face_size)
                        (x, y, w, h) = cropped
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                        cv2.imshow('Faces', frame)
                        imgfile = os.path.basename(video_file).replace(".","_") +"_"+ str(frame_index) + ".png"
                        imgfile = os.path.join(save_folder, imgfile)
                        cv2.imwrite(imgfile, face_img)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()


def main():
    resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                                pooling='avg')  # pooling: None, avg or max
    def image2x(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)  # or version=2
        return x

    def cal_mean_feature(image_folder):
        face_images = list(glob.iglob(os.path.join(image_folder, '*.*')))

        def chunks(l, n):
            """Yield successive n-sized chunks from l."""
            for i in range(0, len(l), n):
                yield l[i:i+n]

        batch_size = 32
        face_images_chunks = chunks(face_images, batch_size)
        fvecs = None
        for face_images_chunk in face_images_chunks:
            images = np.concatenate([image2x(face_image) for face_image in face_images_chunk])
            batch_fvecs = resnet50_features.predict(images)
            if fvecs is None:
                fvecs = batch_fvecs
            else:
                fvecs = np.append(fvecs, batch_fvecs, axis=0)
        return np.array(fvecs).sum(axis=0) / len(fvecs)

    FACE_IMAGES_FOLDER = "./data/face_images"
    VIDEOS_FOLDER = "../data/train/video"
    extractor = FaceExtractor()
    folders = list(glob.iglob(os.path.join(VIDEOS_FOLDER, '*')))
    os.makedirs(FACE_IMAGES_FOLDER, exist_ok=True)
    names = [os.path.basename(folder) for folder in folders]
    for i, folder in enumerate(folders):
        name = names[i]
        videos = list(glob.iglob(os.path.join(folder, '*.*')))
        save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
        print("Save folder:", save_folder)
        os.makedirs(save_folder, exist_ok=True)
        for video in videos:
            extractor.extract_faces(video, save_folder)

    precompute_features = []
    for i, folder in enumerate(folders):
        name = names[i]
        save_folder = os.path.join(FACE_IMAGES_FOLDER, name)
        mean_features = cal_mean_feature(image_folder=save_folder)
        precompute_features.append({"name": name, "features": mean_features})
    pickle_stuff("./data/precompute_features.pickle", precompute_features)


if __name__ == "__main__":
    main()

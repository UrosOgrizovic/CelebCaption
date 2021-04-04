import cv2
import os
import glob
import pickle
from keras_vggface.vggface import VGGFace
import numpy as np
from keras.preprocessing import image
from keras_vggface import utils
import face_helpers
import video_constants as const


class FaceExtractor:
    """
    Singleton class to extract face images from video files
    """
    HAAR_PATH = ".\\pretrained_models\\haarcascade_frontalface_alt.xml"

    def __init__(self, face_size=224):
        self.face_size = face_size

    def extract_faces(self, video_file, save_folder):
        face_cascade = cv2.CascadeClassifier(self.HAAR_PATH)

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
            frame_index += 1
            status, frame = cap.read()
            if not status:
                # video over
                break
            if frame_index % face_detection_period == 0:
                gray, faces = face_helpers.instantiate_gray_and_faces(frame, face_cascade)
                # only keep the biggest face as the main subject
                face = None
                if len(faces) > 1:  # Get the largest face as main face
                    # area = w * h
                    face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))
                elif len(faces) == 1:
                    face = faces[0]
                if face is not None:
                    face_img, cropped = face_helpers.crop_face(frame, face, margin=40, size=self.face_size)
                    (x, y, w, h) = cropped
                    imgfile = os.path.basename(video_file).replace(".","_") +"_"+ str(frame_index) + ".png"
                    imgfile = os.path.join(save_folder, imgfile)
                    cv2.imwrite(imgfile, face_img)
            if cv2.waitKey(5) == 27:  # ESC key press
                break
            if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                # video over
                break
        # When everything is done, release the capture
        cap.release()
        cv2.destroyAllWindows()


def image2x(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1)  # or version=2
    return x


def cal_mean_feature(image_folder, resnet50_features):
    face_images = list(glob.iglob(os.path.join(image_folder, '*.*')))

    batch_size = 32
    face_images_chunks = face_images_generator(face_images, batch_size)
    feature_vectors = None
    for face_images_chunk in face_images_chunks:
        images = np.concatenate([image2x(face_image) for face_image in face_images_chunk])
        batch_feature_vectors = resnet50_features.predict(images)
        if feature_vectors is None:
            feature_vectors = batch_feature_vectors
        else:
            feature_vectors = np.append(feature_vectors, batch_feature_vectors, axis=0)
    return np.array(feature_vectors).sum(axis=0) / len(feature_vectors)


def face_images_generator(face_imgs, batch_size):
    """Yield successive batch sized chunks from face_imgs."""
    for i in range(0, len(face_imgs), batch_size):
        yield face_imgs[i:i+batch_size]


def main():
    resnet50_features = VGGFace(model='resnet50', include_top=False,
                                input_shape=(224, 224, 3),
                                pooling='avg')  # pooling: None, avg or max

    FACE_IMAGES_FOLDER = "./data/face_images"
    VIDEOS_FOLDER = "./data/video"
    extractor = FaceExtractor()
    folders = list(glob.iglob(os.path.join(VIDEOS_FOLDER, '*')))
    os.makedirs(FACE_IMAGES_FOLDER, exist_ok=True)
    precompute_features = []
    for i, folder in enumerate(folders):
        folder_name = os.path.basename(folder)
        videos = list(glob.iglob(os.path.join(folder, '*.*')))
        save_folder = create_save_folder(FACE_IMAGES_FOLDER, folder_name)
        for video in videos:
            extractor.extract_faces(video, save_folder)

        mean_feature = cal_mean_feature(save_folder, resnet50_features)
        precompute_features.append({"name": folder_name, "features": mean_feature})
    pickle_stuff("./data/precompute_features.pickle", precompute_features)


def create_save_folder(base_folder, folder_name):
    save_folder = os.path.join(base_folder, folder_name)
    os.makedirs(save_folder, exist_ok=True)
    return save_folder


def pickle_stuff(filename, stuff):
    save_stuff = open(filename, "wb")
    pickle.dump(stuff, save_stuff)
    save_stuff.close()


if __name__ == "__main__":
    main()

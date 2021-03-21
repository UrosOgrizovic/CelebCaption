import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer
import time
import speech_recognition as sr
import ffmpeg
import os
import helpers
import constants
from keras.models import load_model
from preprocess_audio import process_audio
import scipy.io.wavfile
from csv_preprocess import read_csv
import torchaudio
torchaudio.set_audio_backend("soundfile")
import wget
from speechbrain.pretrained import SepformerSeparation as separator
import sounddevice as sd
import pickle


def PlayVideo(video_path):
    player = MediaPlayer(video_path)

    while True:
        frame, val = player.get_frame()
        if val == 'eof':
            break
        if frame is not None:
            image, pts = frame
            w, h = image.get_size()

            # convert to array width, height
            img = np.asarray(image.to_bytearray()[0]).reshape(h,w,3)

            # convert RGB to BGR because `cv2` need it to display it
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            time.sleep(val)
            cv2.imshow('video', img)

            if cv2.waitKey(1) & 0xff == ord('q'):
                break

    cv2.destroyAllWindows()
    player.close_player()


def extract_audio_from_video(video_path):
    helpers.create_folder(constants.AUDIO_FOLDER_PATH)
    mp3_path = os.path.join(constants.AUDIO_FOLDER_PATH, "speech.mp3")
    wav_path = os.path.join(constants.AUDIO_FOLDER_PATH, "speech.wav")

    ''' notes: 1. os.system() demands double quotes around paths containing spaces
    2. The -y flag forces overwriting the existing file. Remove the flag for a y/N query.
    3. -af "pan=mono|FC=FR" converts from stereo to mono (the right channel is used). This
    is necessary in order for process_audio() to work correctly.
    '''
    command2mp3 = f'ffmpeg -i "{video_path}" -y -af "pan=mono|FC=FR" "{mp3_path}"'
    command2wav = f'ffmpeg -i "{mp3_path}" -y "{wav_path}"'
    os.system(command2mp3)  # execute command
    os.system(command2wav)
    return wav_path


def extract_text_from_audio(audio_path):
    """Performs speech recognition

    Args:
        audio_path (string): .wav file path

    Returns:
        string: recognized text
    """
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    text = r.recognize_google(audio)
    return text


def predict_vggvox(model_weights_path, audio_path):
    vggvox_model = load_model(model_weights_path)
    _, wave = scipy.io.wavfile.read(audio_path, mmap=True)

    processed = process_audio(wave)
    assert len(processed.shape) == 3
    assert processed.shape[0] == 512
    assert processed.shape[2] == 1
    # there's only one sample in the batch, so add 1 as the first dim
    processed = np.expand_dims(processed, axis=0)

    prediction = vggvox_model.predict(processed).flatten()
    print(prediction, np.argmax(prediction), prediction.shape)
    print(prediction[np.argmax(prediction)], read_csv('vox1_meta.csv')[np.argmax(prediction)])


def speaker_diarization(audio_path):
    model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')
    mix, fs = torchaudio.load(audio_path)
    # resampling because the model being used was trained on 8KHz data.
    resampler = torchaudio.transforms.Resample(fs, 8000)
    mix = resampler(mix)
    est_sources = model.separate_batch(mix)
    est_sources = est_sources[0]    # strip batch dimension

    pickle.dump(est_sources, open('est_sources.pkl', 'wb'))
    return est_sources.shape[1] # return the number of speakers


def play_diarized_sounds(num_speakers):
    fs = 8000
    est_sources = pickle.load(open('est_sources.pkl', 'rb'))
    for i in range(num_speakers):
        sd.play(est_sources[:, i], fs)
        sd.wait()


if __name__ == '__main__':
    video_path = "data/test/video/Bradley Cooper, Lady Gaga/Bradley Cooper, Lady Gaga.mp4"
    # PlayVideo(video_path)
    audio_path = extract_audio_from_video(video_path)
    # text = extract_text_from_audio(audio_path)
    # print(text)
    num_speakers = speaker_diarization(audio_path)
    play_diarized_sounds(num_speakers)
    # model_path = "weights\with-augmentation.hdf5"
    # predict_vggvox(model_path, audio_path)
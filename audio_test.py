import cv2
import numpy as np
from ffpyplayer.player import MediaPlayer
import time
import speech_recognition as sr
import ffmpeg
import os
import helpers
import constants
from tensorflow.keras.models import load_model
from preprocess_audio import process_audio
import scipy.io.wavfile
from csv_preprocess import read_csv
import torchaudio
torchaudio.set_audio_backend("soundfile")   # change depending on OS, "soundfile" for Windows
import wget
from speechbrain.pretrained import SepformerSeparation as separator
import sounddevice as sd
import pickle
import tensorflow as tf
import torch
import datetime


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


def speaker_diarization(audio_path):
    model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir='pretrained_models/sepformer-wsj02mix')
    mix, fs = torchaudio.load(audio_path)
    # resampling because the model being used was trained on 8KHz data.
    resampler = torchaudio.transforms.Resample(fs, 8000)
    mix = resampler(mix)
    est_sources = model.separate_batch(mix)
    est_sources = est_sources[0]    # strip batch dimension

    pickle.dump(est_sources, open('est_sources.pkl', 'wb'))
    num_speakers = est_sources.shape[1]
    audio_paths = []
    for i in range(num_speakers):
        path = f'data/test/audio/diarized/speaker_{i}.wav'
        torchaudio.save(path, torch.unsqueeze(est_sources[:, i], 0), 8000, encoding="PCM_S")
        audio_paths.append(path)
    return audio_paths


def extract_text_from_audio(audio_path, segment_num, start_time, subs_file_name):
    """Performs speech recognition

    Args:
        audio_path (string): .wav file path

    Returns:
        string: recognized text divided into segments
        no longer than 15 words
    """
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    text = r.recognize_google(audio)
    segments = []
    split_txt = text.split(" ")
    length = len(split_txt)
    if length > 15:
        # 15 words in each subtitle segment seems reasonable
        for i in range(15, length, 15):
            segments.append(" ".join(split_txt[i-15:i]))
            if i > length - 15: # last segment
                segments.append(" ".join(split_txt[i:]))
    else:
        segments = " ".join(segments)

    for seg in segments:
        segment_num += 1
        end_time = time_addition(seg, start_time, segment_num, subs_file_name)
        start_time = end_time
        # add half a second so that subs don't overlap
        # start_time += datetime.timedelta(0, 0.5)


    return segments, start_time


def time_addition(sentence, current_time, segment_num, subs_file_name):
    """ Calculates the number of words in a string and adds them in seconds to timestamp"""

    time_add = (len(sentence.split()))*0.35 #takes an average of 1 second to read a word so number of words = added seconds
    end_time = current_time + datetime.timedelta(0, time_add) #add the time in seconds
    str_current_time = str(current_time.time()) #convert the timestamps to string
    str_current_time = str_current_time.replace(".", ",")   # change fractional separator
    str_end_time = str(end_time.time()) #if you try and write to file without converting you will get type error 'datetime.time'
    str_end_time = str_end_time.replace(".", ",") # change fractional separator

    #now we to add to the .srt
    #append the time according to the caclulation of number of words per sentence
    #making sure we are consistent with appropriate formatting of the text file
    with open(subs_file_name, "a") as f:
        f.write(str(segment_num))
        f.write("\n")
        f.write(str_current_time)
        f.write(" --> ")
        f.write(str_end_time)
        f.write("\n")
        f.write(sentence)
        f.write("\n")
        f.write("\n")
    return end_time


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
    name = read_csv('vox1_meta.csv')[np.argmax(prediction)]
    name = name.replace('_', ' ')
    prob = prediction[np.argmax(prediction)]

    return name, round(prob*100, 2)


def play_audio_file(audio_path):
    mix, fs = torchaudio.load(audio_path)
    sd.play(mix.squeeze(), fs)
    sd.wait()


if __name__ == '__main__':
    video_path = "data/test/video/Bradley Cooper, Lady Gaga/Bradley Cooper, Lady Gaga.mp4"
    model_path = "weights\with-augmentation.hdf5"
    # PlayVideo(video_path)
    # audio_path = extract_audio_from_video(video_path)
    # audio_paths = speaker_diarization(audio_path)
    audio_paths = ['data/test/audio/diarized/speaker_0.wav', 'data/test/audio/diarized/speaker_1.wav']

    subs_file_name = "unsynchronized.srt"
    open(subs_file_name, "w").close()  # clear file contents
    segment_num = 0
    start_time = datetime.datetime(100,1,1,0,0,1, 10**4)
    for i in range(len(audio_paths)-1, -1, -1):
        # name, prob = predict_vggvox(model_path, audio_paths[i])
        # print(f"Speaker #{i} is {name} with {round(prob*100, 2)}% confidence")
        segments, start_time = extract_text_from_audio(audio_paths[i], segment_num, start_time, subs_file_name)
        segment_num += len(segments)
        # play_audio_file(audio_paths[i])

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
import wget
from speechbrain.pretrained import SepformerSeparation as separator
import sounddevice as sd
import pickle
import tensorflow as tf
import datetime
import auditok


def media_play_video(video_path):
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
    2. The -y flag overwrites the existing file. Remove the flag for a y/N query.
    3. -af "pan=mono|FC=FR" converts from stereo to mono (the right channel is used). This
    is necessary in order for process_audio() to work correctly.
    '''
    command2mp3 = f'ffmpeg -i "{video_path}" -y -af "pan=mono|FC=FR" "{mp3_path}"'
    command2wav = f'ffmpeg -i "{mp3_path}" -y "{wav_path}"'
    os.system(command2mp3)  # execute command
    os.system(command2wav)
    return mp3_path, wav_path


def add_audio_to_video(input_video_path, mp3_path, output_video_path):
    command = f'ffmpeg -i {input_video_path} -i {mp3_path} -y -c copy -map 0:v:0 -map 1:a:0 {output_video_path}'
    os.system(command)


def extract_text_from_audio(audio_path, subs_file_name, speaker_name, start_time=datetime.datetime(1,1,1,0,0,0,0),
                            end_time=datetime.datetime(1,1,1,0,0,1,0), segment_num=0):
    """Performs speech recognition

    Args:
        audio_path (string): .wav file path

    Returns:
        string: recognized text divided into text segments
        no longer than 15 words
    """
    r = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio = r.record(source)
    text = r.recognize_google(audio)
    print(f'Text: {text}')
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
        segments = [text]   # 1 segment is enough

    for seg in segments:
        segment_num += 1
        end_time = write_subs_to_file(seg, start_time, end_time, segment_num, subs_file_name, speaker_name)
        start_time = end_time

    return segments, start_time


def write_subs_to_file(segment, start_time, end_time, segment_num, subs_file_name, speaker_name):
    str_start_time = str(start_time.time()).replace(".", ",")   # change fractional separator
    str_end_time = str(end_time.time()).replace(".", ",") #if you try and write to file without converting you will get type error 'datetime.time'

    # add to the .srt
    with open(subs_file_name, "a") as f:
        f.write(str(segment_num))
        f.write("\n")
        f.write(str_start_time)
        f.write(" --> ")
        f.write(str_end_time)
        f.write("\n")
        f.write(f'{speaker_name}: {segment}')
        f.write("\n")
        f.write("\n")
    return end_time


def predict_vggvox(vggvox_model, audio_path):
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


if __name__ == '__main__':
    # original_video_path = "data/test/video/cooper gaga.mp4"
    original_video_path = "data/test/video/fallon stiller.mp4"
    processed_video_path = 'processed_video.mp4'
    output_processed_video_path = 'processed_video_with_audio.mp4'
    model_weights_path = "weights\with-augmentation.hdf5"
    subs_file_name = "unsynchronized.srt"
    open(subs_file_name, "w").close()  # clear file contents
    vggvox_model = load_model(model_weights_path)

    mp3_path, wav_path = extract_audio_from_video(original_video_path)
    add_audio_to_video(processed_video_path, mp3_path, output_processed_video_path)

    # split returns a generator of AudioRegion objects
    audio_regions = auditok.split(
        wav_path,
        min_dur=2,     # minimum duration of a valid audio event in seconds
        max_dur=6,       # maximum duration of an event
        max_silence=1, # maximum duration of tolerated continuous silence within an event
        energy_threshold=10 # threshold of detection
    )

    for segment_num, r in enumerate(audio_regions):
        # ignore tiny leftovers at the end of the audio
        if r.duration >= 2:
            # Regions returned by `split` have 'start' and 'end' metadata fields
            region_file_name = f"region_{int(r.meta.start)}-{int(r.meta.end)}.wav"
            r.save(region_file_name)
            start_time = datetime.datetime(1,1,1,0,0,int(r.meta.start),0)
            end_time = datetime.datetime(1,1,1,0,0,int(r.meta.end),0)
            speaker_name, prob = predict_vggvox(vggvox_model, region_file_name)
            extract_text_from_audio(region_file_name, subs_file_name, speaker_name,
                                    start_time, end_time, segment_num)


    # media_play_video(output_processed_video_path)

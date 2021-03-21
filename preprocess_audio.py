import numpy
import decimal
import math
from scipy.signal import lfilter

# Preprocessing is based on https://github.com/linhdvu14/vggvox-speaker-identification
# with some fixes
PREEMPHASIS_ALPHA = 0.97
FRAME_LEN = 0.025
FRAME_STEP = 0.01
NUM_FFT = 512
SAMPLE_RATE = 16e3
MAX_SEC_TEST = 10
BUCKET_STEP_SEC = 1
CLIP_SIZE = 3 + 0.5 # 3 seconds and 0.5 to have some additional offset


def build_buckets():
    """Generates array of buckets (lengths to clip fft output)"""
    buckets = {}
    frames_per_sec = int(1 / FRAME_STEP)
    end_frame = int(MAX_SEC_TEST * frames_per_sec)
    step_frame = int(BUCKET_STEP_SEC * frames_per_sec)
    for i in range(0, end_frame + 1, step_frame):
        s = i
        s = numpy.floor((s - 7 + 2) / 2) + 1  # conv1
        s = numpy.floor((s - 3) / 2) + 1  # mpool1
        s = numpy.floor((s - 5 + 2) / 2) + 1  # conv2
        s = numpy.floor((s - 3) / 2) + 1  # mpool2
        s = numpy.floor((s - 3 + 2) / 1) + 1  # conv3
        s = numpy.floor((s - 3 + 2) / 1) + 1  # conv4
        s = numpy.floor((s - 3 + 2) / 1) + 1  # conv5
        s = numpy.floor((s - 3) / 2) + 1  # mpool5
        s = numpy.floor((s - 1) / 1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)

    return buckets


BUCKETS = build_buckets()


def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def rolling_window(a, window, step=1):
    # http://ellisvalentiner.com/post/2017-03-21-np-strides-trick
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return numpy.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)[::step]


def framesig(sig, frame_len, frame_step, winfunc=lambda x: numpy.ones((x,)), stride_trick=True):
    """Frame a signal into overlapping frames.
    :param sig: the audio signal to frame.
    :param frame_len: length of each frame measured in samples.
    :param frame_step: number of samples after the start of the previous frame that the next frame should begin.
    :param winfunc: the analysis window to apply to each frame. By default no window is applied.
    :param stride_trick: use stride trick to compute the rolling window and window multiplication faster
    :returns: an array of frames. Size is NUMFRAMES by frame_len.
    """
    slen = len(sig)
    frame_len = int(round_half_up(frame_len))
    frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = int(math.floor(1 + (1.0 * slen - frame_len) / frame_step))  # LV
    padsignal = sig
    if stride_trick:
        win = winfunc(frame_len)
        frames = rolling_window(padsignal, window=frame_len, step=frame_step)
    else:
        indices = numpy.tile(numpy.arange(0, frame_len), (numframes, 1)) + numpy.tile(
            numpy.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
        indices = numpy.array(indices, dtype=numpy.int32)
        frames = padsignal[indices]
        win = numpy.tile(winfunc(frame_len), (numframes, 1))

    return frames * win


def preemphasis(signal, coeff=0.95):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :returns: the filtered signal.
    """
    return numpy.append(signal[0], signal[1:] - coeff * signal[:-1])


# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m
def rm_dc_n_dither(sin, sample_rate):
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print("Sample rate must be 16kHz or 8kHz only")
        alpha = 0.99
    sin = lfilter([1, -1], [1, -alpha], sin)
    dither = numpy.ones(shape=(len(sin)))
    spow = numpy.std(sin, ddof=1)
    sout = sin + 1e-6 * spow * dither
    return sout


def clip_signal(signal, limit):
    limit = int(limit)
    start = numpy.random.randint(0, signal.shape[0] - limit)
    return signal[start:start + limit]


def process_audio(signal, clip=False):
    """Outputs clipped to 3 seconds for training/validation, normalized spectrogram
    Most of the clipping is done before calculating fft for sake of speed"""
    if clip:
        signal = clip_signal(signal, CLIP_SIZE * SAMPLE_RATE)
    if max(abs(signal)) <= 1:
        signal *= 2 ** 15
    # Process signal to get FFT spectrum
    signal = rm_dc_n_dither(signal, SAMPLE_RATE)
    signal = preemphasis(signal, coeff=PREEMPHASIS_ALPHA)
    frames = framesig(signal,
                      frame_len=FRAME_LEN * SAMPLE_RATE,
                      frame_step=FRAME_STEP * SAMPLE_RATE,
                      winfunc=numpy.hamming)
    fft = abs(numpy.fft.fft(frames, n=NUM_FFT))
    mu = numpy.mean(fft, axis=0)
    std = numpy.std(fft, axis=0, ddof=1)
    fft = (fft - mu) / std
    fft = fft.T
    rsize = max(k for k in BUCKETS if k <= fft.shape[1])
    rstart = round((fft.shape[1] - rsize) / 2) - 1
    if rstart <= 0:
        out = fft
    else:
        out = fft[:, rstart:rsize + rstart]

    return out.reshape(*out.shape, 1)
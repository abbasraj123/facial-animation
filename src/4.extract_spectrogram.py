import math
import pathlib as plb
import librosa
import csv
import numpy as np

from extract_feature import write_csv, get_fps, extract_one_frame_data
from project_paths import paths, parse_data_root, ensure_dirs

FREQ_DIM = 128
TIME_DIM = 32
NFFT = FREQ_DIM*2

nFrameSize = (TIME_DIM - 3) * FREQ_DIM + NFFT

def extract_one_file(videofile, audiofile):
    print (" --- " + audiofile)
    # get video FPS
    nFrames, fps = get_fps(videofile)
    # load audio
    data, sr = librosa.load(audiofile, sr=44100) # data is np.float32
    # number of audio samples per video frame
    nSamPerFrame = int(math.floor(float(sr) / fps))
    # number of samples per 20ms
    nSamPerFFTWindow = NFFT #int(math.ceil(float(sr) * 0.02))
    # number of samples per step 8ms
    nSamPerStep = FREQ_DIM #int(math.floor(float(sr) * 0.008))
    # number of steps per frame
    nStepsPerFrame = TIME_DIM #int(math.floor(float(nSamPerFrame) / float(nSamPerStep)))
    # real frame size
    nFrameSize = (nStepsPerFrame - 1) * nSamPerStep + nSamPerFFTWindow
    # initial position in the sound stream
    # initPos negative means we need zero padding at the front.
    curPos = nSamPerFrame - nFrameSize
    dbspecs = []
    for f in range(0,nFrames):
        frameData, nextPos = extract_one_frame_data(data, curPos, nFrameSize, nSamPerFrame)
        curPos = nextPos
        # spectrogram transform
        FD = librosa.core.stft(y=frameData, n_fft=NFFT, hop_length=FREQ_DIM)
        FD, phase = librosa.magphase(FD)
        DB = librosa.core.amplitude_to_db(FD, ref=np.max)
        # scale dB-spectrogram in [0,1]
        DB = np.divide(np.absolute(DB), 80.0)
        # remove the last row
        newDB = DB[0:-1,:]
        if newDB.shape[1] > TIME_DIM:
            newDB = newDB[:, :TIME_DIM]
        elif newDB.shape[1] < TIME_DIM:
            padded = np.zeros((FREQ_DIM, TIME_DIM), dtype=newDB.dtype)
            padded[:, :newDB.shape[1]] = newDB
            newDB = padded
        # store
        dbspecs.append(newDB.flatten().tolist())
    return dbspecs

def process_all(video_root, audio_root, feat_root):
    video_dir = plb.Path(video_root)
    feat_dir = plb.Path(feat_root)
    ensure_dirs(feat_dir)
    for actor in video_dir.iterdir():
        if actor.is_file() and actor.suffix.lower() == ".mp4":
            video_files = [actor]
            actor_name = "default"
        elif actor.is_dir():
            video_files = list(actor.iterdir())
            actor_name = actor.name
        else:
            continue
        for video_file in video_files:
            if video_file.suffix.lower() != ".mp4":
                continue
            seq_dir = plb.Path(feat_root) / actor_name / video_file.stem
            ensure_dirs(seq_dir)
            video_path = str(video_file)
            audio_path = str(plb.Path(audio_root) / actor_name / (video_file.stem + ".wav"))
            if actor_name == "default" and not plb.Path(audio_path).exists():
                audio_path = str(plb.Path(audio_root) / (video_file.stem + ".wav"))
            dbspecs = extract_one_file(video_path, audio_path)
            feature_path = str(seq_dir / "dbspectrogram.csv")
            write_csv(feature_path, dbspecs)

if __name__ == "__main__":
    data_root = parse_data_root("Extract dB spectrogram features")
    pipeline_paths = paths(data_root)
    process_all(pipeline_paths["video"], pipeline_paths["speech"], pipeline_paths["features"])

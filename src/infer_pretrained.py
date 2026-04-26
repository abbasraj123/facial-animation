import pathlib as plb
import subprocess as sup
import importlib.util
import os
import sys
import argparse

import cv2
import librosa
import numpy as np

from project_paths import paths, add_data_root_arg, ensure_dirs, get_data_root
from extract_feature import write_csv

FREQ_DIM = 128
TIME_DIM = 32
MODEL_INPUT_SIZE = FREQ_DIM * TIME_DIM


def prepare_cntk_dll_path():
    cntk_libs = plb.Path(sys.prefix) / "Lib" / "site-packages" / "cntk" / "libs"
    if cntk_libs.exists():
        os.environ["PATH"] = str(cntk_libs) + os.pathsep + os.environ.get("PATH", "")


prepare_cntk_dll_path()
import cntk as C


def load_numbered_module(filename, module_name):
    script_path = plb.Path(__file__).resolve().parent / filename
    spec = importlib.util.spec_from_file_location(module_name, str(script_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spectrogram_module = load_numbered_module("4.extract_spectrogram.py", "numbered_extract_spectrogram")
eval_module = load_numbered_module("9.eval_speech.py", "numbered_eval_speech")
extract_one_file = spectrogram_module.extract_one_file
estimate_one_audio_seq = eval_module.estimate_one_audio_seq


SUPPORTED_VIDEO_EXTENSIONS = [".mp4"]
SUPPORTED_AUDIO_EXTENSIONS = [".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"]


def find_input_media(data_root, video_root, only_media=None):
    roots = [plb.Path(data_root), plb.Path(video_root)]
    seen = set()
    only_media_name = plb.Path(only_media).name if only_media else None
    for root in roots:
        if not root.exists():
            continue
        for item in root.iterdir():
            if item.is_file() and item.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS + SUPPORTED_AUDIO_EXTENSIONS and item not in seen:
                if only_media_name and item.name != only_media_name:
                    continue
                seen.add(item)
                yield "default", item
            elif item.is_dir():
                for media_file in item.iterdir():
                    if media_file.is_file() and media_file.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS + SUPPORTED_AUDIO_EXTENSIONS and media_file not in seen:
                        if only_media_name and media_file.name != only_media_name:
                            continue
                        seen.add(media_file)
                        yield item.name, media_file


def extract_audio(video_file, audio_file):
    ensure_dirs(plb.Path(audio_file).parent)
    command = 'ffmpeg -i "{}" -y -ab 160k -ac 2 -ar 44100 -vn "{}" -loglevel quiet'.format(video_file, audio_file)
    sup.check_call(command, shell=True)


def convert_audio(audio_source, audio_file):
    ensure_dirs(plb.Path(audio_file).parent)
    if plb.Path(audio_source).suffix.lower() == ".wav" and plb.Path(audio_source).resolve() == plb.Path(audio_file).resolve():
        return
    command = 'ffmpeg -i "{}" -y -ab 160k -ac 2 -ar 44100 "{}" -loglevel quiet'.format(audio_source, audio_file)
    sup.check_call(command, shell=True)


def extract_frames(video_file, max_frames=None):
    cap = cv2.VideoCapture(str(video_file))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.resize(frame, (100, 100), interpolation=cv2.INTER_CUBIC))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return np.array(frames, dtype=np.float32)


def resolve_model(pipeline_paths):
    model_file = pipeline_paths["models"] / "model_audio2exp_2019-03-24-21-03.dnn"
    if model_file.exists():
        return model_file
    return plb.Path(__file__).resolve().parents[1] / "Outputs" / "model_audio2exp_2019-03-24-21-03" / "model_audio2exp_2019-03-24-21-03.dnn"


def normalize_audio_features(audio):
    audio = np.atleast_2d(audio).astype(np.float32)
    if audio.shape[1] == MODEL_INPUT_SIZE:
        return audio
    if audio.shape[1] < MODEL_INPUT_SIZE:
        padded = np.zeros((audio.shape[0], MODEL_INPUT_SIZE), dtype=np.float32)
        padded[:, :audio.shape[1]] = audio
        return padded
    return audio[:, :MODEL_INPUT_SIZE]


def extract_audio_only_spectrogram(audiofile, fps=15):
    data, sr = librosa.load(str(audiofile), sr=44100)
    n_frames = int(np.ceil(float(len(data)) / float(sr) * fps))
    n_sam_per_frame = int(np.floor(float(sr) / fps))
    n_fft = FREQ_DIM * 2
    n_frame_size = (TIME_DIM - 1) * FREQ_DIM + n_fft
    cur_pos = n_sam_per_frame - n_frame_size
    dbspecs = []
    for _ in range(n_frames):
        frame_data = np.zeros(n_frame_size, dtype=np.float32)
        if cur_pos < 0:
            start_pos = -cur_pos
            chunk = data[0:(cur_pos + n_frame_size)]
            frame_data[start_pos:start_pos + len(chunk)] = chunk
        else:
            chunk = data[cur_pos:(cur_pos + n_frame_size)]
            frame_data[:len(chunk)] = chunk
        cur_pos += n_sam_per_frame
        fd = librosa.core.stft(y=frame_data, n_fft=n_fft, hop_length=FREQ_DIM)
        fd, _ = librosa.magphase(fd)
        db = librosa.core.amplitude_to_db(fd, ref=np.max)
        db = np.divide(np.absolute(db), 80.0)
        new_db = db[0:-1, :]
        if new_db.shape[1] > TIME_DIM:
            new_db = new_db[:, :TIME_DIM]
        elif new_db.shape[1] < TIME_DIM:
            padded = np.zeros((FREQ_DIM, TIME_DIM), dtype=new_db.dtype)
            padded[:, :new_db.shape[1]] = new_db
            new_db = padded
        dbspecs.append(new_db.flatten().tolist())
    return dbspecs


def infer_one(model, media_file, audio_file, feature_file, output_dir):
    ensure_dirs(feature_file.parent, output_dir)
    is_video = plb.Path(media_file).suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS
    if is_video:
        if not audio_file.exists():
            extract_audio(media_file, audio_file)
        dbspecs = extract_one_file(str(media_file), str(audio_file))
    else:
        if not audio_file.exists():
            convert_audio(media_file, audio_file)
        dbspecs = extract_audio_only_spectrogram(audio_file)
    write_csv(str(feature_file), dbspecs)

    audio = normalize_audio_features(np.loadtxt(str(feature_file), dtype=np.float32, delimiter=","))
    audio_seq = np.reshape(audio, (audio.shape[0], 1, FREQ_DIM, TIME_DIM))
    fake = estimate_one_audio_seq(model, audio_seq)
    if fake.shape[1] == 49:
        fake = fake[:, 3:]
    if fake.shape[1] != 46:
        raise ValueError("unsupported output of audio model")

    if is_video:
        frames = extract_frames(media_file, max_frames=fake.shape[0])
    else:
        frames = np.zeros((fake.shape[0], 100, 100, 3), dtype=np.float32)
    if frames.shape[0] != fake.shape[0]:
        frames = np.zeros((fake.shape[0], 100, 100, 3), dtype=np.float32)

    np.save(str(output_dir / "fake.npy"), fake.astype(np.float32))
    np.save(str(output_dir / "real.npy"), np.zeros_like(fake, dtype=np.float32))
    np.save(str(output_dir / "img.npy"), frames.astype(np.float32))
    print("Saved inference output to " + str(output_dir))


def main():
    parser = argparse.ArgumentParser(description="Run pretrained model inference without RAVDESS labels")
    add_data_root_arg(parser)
    parser.add_argument("--only-video", default=None, help="Process only this .mp4 filename")
    parser.add_argument("--only-media", default=None, help="Process only this video or audio filename")
    parser.add_argument("--only-audio", default=None, help="Process only this audio filename")
    args = parser.parse_args()
    data_root = get_data_root(args.data_root)
    pipeline_paths = paths(data_root)
    ensure_dirs(
        pipeline_paths["speech"],
        pipeline_paths["features"],
        pipeline_paths["outputs"] / "npy_output",
        pipeline_paths["models"],
    )

    model_file = resolve_model(pipeline_paths)
    if not model_file.exists():
        raise IOError("pretrained model not found: " + str(model_file))
    model = C.load_model(str(model_file))

    found = False
    only_media = args.only_media or args.only_audio or args.only_video
    for actor_name, media_file in find_input_media(pipeline_paths["root"], pipeline_paths["video"], only_media):
        found = True
        seq_name = media_file.stem
        audio_file = pipeline_paths["speech"] / actor_name / (seq_name + ".wav")
        feature_file = pipeline_paths["features"] / actor_name / seq_name / "dbspectrogram.csv"
        output_dir = pipeline_paths["outputs"] / "npy_output" / actor_name / seq_name
        infer_one(model, media_file, audio_file, feature_file, output_dir)

    if not found:
        raise IOError("No matching video/audio files found in " + str(pipeline_paths["root"]) + " or " + str(pipeline_paths["video"]))


if __name__ == "__main__":
    main()

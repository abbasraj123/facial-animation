import pathlib as plb
import subprocess as sup
import importlib.util
import os
import sys
import argparse

import cv2
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


def find_input_videos(data_root, video_root, only_video=None):
    roots = [plb.Path(data_root), plb.Path(video_root)]
    seen = set()
    only_video_name = plb.Path(only_video).name if only_video else None
    for root in roots:
        if not root.exists():
            continue
        for item in root.iterdir():
            if item.is_file() and item.suffix.lower() == ".mp4" and item not in seen:
                if only_video_name and item.name != only_video_name:
                    continue
                seen.add(item)
                yield "default", item
            elif item.is_dir():
                for video_file in item.iterdir():
                    if video_file.is_file() and video_file.suffix.lower() == ".mp4" and video_file not in seen:
                        if only_video_name and video_file.name != only_video_name:
                            continue
                        seen.add(video_file)
                        yield item.name, video_file


def extract_audio(video_file, audio_file):
    ensure_dirs(plb.Path(audio_file).parent)
    command = 'ffmpeg -i "{}" -y -ab 160k -ac 2 -ar 44100 -vn "{}" -loglevel quiet'.format(video_file, audio_file)
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


def infer_one(model, video_file, audio_file, feature_file, output_dir):
    ensure_dirs(feature_file.parent, output_dir)
    if not audio_file.exists():
        extract_audio(video_file, audio_file)
    dbspecs = extract_one_file(str(video_file), str(audio_file))
    write_csv(str(feature_file), dbspecs)

    audio = normalize_audio_features(np.loadtxt(str(feature_file), dtype=np.float32, delimiter=","))
    audio_seq = np.reshape(audio, (audio.shape[0], 1, FREQ_DIM, TIME_DIM))
    fake = estimate_one_audio_seq(model, audio_seq)
    if fake.shape[1] == 49:
        fake = fake[:, 3:]
    if fake.shape[1] != 46:
        raise ValueError("unsupported output of audio model")

    frames = extract_frames(video_file, max_frames=fake.shape[0])
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
    for actor_name, video_file in find_input_videos(pipeline_paths["root"], pipeline_paths["video"], args.only_video):
        found = True
        seq_name = video_file.stem
        audio_file = pipeline_paths["speech"] / actor_name / (seq_name + ".wav")
        feature_file = pipeline_paths["features"] / actor_name / seq_name / "dbspectrogram.csv"
        output_dir = pipeline_paths["outputs"] / "npy_output" / actor_name / seq_name
        infer_one(model, video_file, audio_file, feature_file, output_dir)

    if not found:
        raise IOError("No matching .mp4 files found in " + str(pipeline_paths["root"]) + " or " + str(pipeline_paths["video"]))


if __name__ == "__main__":
    main()

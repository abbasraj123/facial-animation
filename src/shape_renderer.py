import numpy as np
import cv2
from scipy.signal import medfilt
import pathlib as plb
import subprocess as sup
import argparse
import os
from scipy.io import wavfile
import ShapeUtils as SU
from project_paths import paths, add_data_root_arg, ensure_dirs


DEFAULT_EXPRESSION_SCALE = 6.0
DEFAULT_MOUTH_SCALE = 4.0
DEFAULT_AUDIO_MOUTH_STRENGTH = 0.8
MOUTH_EXPRESSION_INDICES = [0, 1, 2, 3, 6, 7, 12, 13, 14, 15, 23, 26, 30, 35]
AUDIO_MOUTH_INDICES = [0, 1, 2, 3, 6]


def visualize_one_audio_seq(visualizer, lf, lr, limg, save_dir):
    ar_dir = plb.Path(save_dir)
    ensure_dirs(ar_dir)
    for i in range(min(len(lf), len(lr), len(limg))):
        ef=lf[i]
        er=lr[i]
        img=limg[i]
        ret = visualizer.visualize(img, er, ef)
        # draw plot
        plot = SU.draw_error_bar_plot(er, ef, (ret.shape[1],200))
        ret = np.concatenate([ret, plot], axis=0)
        save_path = str(ar_dir) + "/result{:06d}.jpg".format(i)
        cv2.imwrite(save_path, ret)
        # can call cv2.imshow() here

def make_video_from_frames(frame_dir, video_file, audio_file=None, fps=15, crop=None, output_size=None):
    frame_dir = plb.Path(frame_dir)
    files = sorted(frame_dir.glob("result*.jpg"))
    if not files:
        return
    first = cv2.imread(str(files[0]))
    if crop:
        x1, y1, x2, y2 = crop
        first = first[y1:y2, x1:x2]
    if output_size:
        first = cv2.resize(first, output_size, interpolation=cv2.INTER_CUBIC)
    height, width, _ = first.shape
    ensure_dirs(plb.Path(video_file).parent)
    video_file = plb.Path(video_file)
    no_audio_file = video_file.with_name(video_file.stem + "_no_audio.mp4")
    writer = cv2.VideoWriter(str(no_audio_file), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for file_path in files:
        frame = cv2.imread(str(file_path))
        if frame is not None:
            if crop:
                x1, y1, x2, y2 = crop
                frame = frame[y1:y2, x1:x2]
            if output_size:
                frame = cv2.resize(frame, output_size, interpolation=cv2.INTER_CUBIC)
            writer.write(frame)
    writer.release()

    if audio_file and plb.Path(audio_file).exists():
        command = 'ffmpeg -y -i "{}" -i "{}" -c:v copy -c:a aac -shortest "{}" -loglevel error'.format(
            no_audio_file, audio_file, video_file
        )
        ret = sup.call(command, shell=True)
        if ret == 0 and no_audio_file.exists():
            no_audio_file.unlink()
    else:
        no_audio_file.replace(video_file)


def scale_expression(values, neutral, scale):
    if scale == 1.0:
        return values
    if np.allclose(neutral, 0.0):
        baseline = np.mean(values, axis=0, keepdims=True)
    else:
        baseline = neutral
    return np.clip(baseline + (values - baseline) * scale, 0.0, 1.0)


def boost_mouth_expression(values, mouth_scale):
    if mouth_scale == 1.0:
        return values
    boosted = np.array(values, copy=True)
    valid_indices = [idx for idx in MOUTH_EXPRESSION_INDICES if idx < boosted.shape[1]]
    mouth = boosted[:, valid_indices]
    baseline = np.mean(mouth, axis=0, keepdims=True)
    boosted[:, valid_indices] = np.clip(baseline + (mouth - baseline) * mouth_scale, 0.0, 1.0)
    return boosted


def audio_energy_envelope(audio_file, frame_count, fps=15):
    if not audio_file or not plb.Path(audio_file).exists() or frame_count < 1:
        return None
    sr, data = wavfile.read(str(audio_file))
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    samples_per_frame = max(1, int(float(sr) / fps))
    envelope = []
    for i in range(frame_count):
        start = i * samples_per_frame
        end = min(len(data), start + samples_per_frame)
        if start >= len(data):
            envelope.append(0.0)
        else:
            chunk = data[start:end]
            envelope.append(float(np.sqrt(np.mean(chunk * chunk))))
    envelope = np.array(envelope, dtype=np.float32)
    if envelope.max() <= 0:
        return None
    low = np.percentile(envelope, 20)
    high = np.percentile(envelope, 95)
    if high <= low:
        high = envelope.max()
    envelope = np.clip((envelope - low) / max(high - low, 1e-6), 0.0, 1.0)
    return envelope


def apply_audio_mouth_drive(values, audio_file, strength):
    if strength <= 0:
        return values
    envelope = audio_energy_envelope(audio_file, values.shape[0])
    if envelope is None:
        return values
    driven = np.array(values, copy=True)
    valid_indices = [idx for idx in AUDIO_MOUTH_INDICES if idx < driven.shape[1]]
    driven[:, valid_indices] = np.clip(driven[:, valid_indices] + envelope[:, None] * strength, 0.0, 1.0)
    return driven


def process_one(
    input_dir,
    frame_output_dir,
    video_output_file,
    video_3d_file,
    audio_file=None,
    expression_scale=DEFAULT_EXPRESSION_SCALE,
    mouth_scale=DEFAULT_MOUTH_SCALE,
    audio_mouth_strength=DEFAULT_AUDIO_MOUTH_STRENGTH,
):
    visualizer = SU.Visualizer()
    path = plb.Path(input_dir)
    lf = np.load(str(path / 'fake.npy'))
    lr = np.load(str(path / 'real.npy'))
    limg = np.load(str(path / 'img.npy'))
    if np.allclose(lf, lr):
        lr = np.zeros_like(lf)
    lf = scale_expression(lf, lr, expression_scale)
    lf = boost_mouth_expression(lf, mouth_scale)
    lf = apply_audio_mouth_drive(lf, audio_file, audio_mouth_strength)
    visualize_one_audio_seq(visualizer, lf, lr, limg, frame_output_dir)
    make_video_from_frames(frame_output_dir, video_output_file, audio_file)
    make_video_from_frames(frame_output_dir, video_3d_file, audio_file, crop=(600, 0, 900, 300), output_size=(600, 600))
    
def process_all(
    data_root=None,
    expression_scale=DEFAULT_EXPRESSION_SCALE,
    mouth_scale=DEFAULT_MOUTH_SCALE,
    audio_mouth_strength=DEFAULT_AUDIO_MOUTH_STRENGTH,
    only_video=None,
):
    pipeline_paths = paths(data_root)
    npy_root = pipeline_paths["outputs"] / "npy_output"
    frame_root = pipeline_paths["outputs"] / "rendered_frames"
    video_root = pipeline_paths["outputs"] / "rendered_videos"
    video_3d_root = pipeline_paths["outputs"] / "rendered_videos_3d"
    only_seq_name = plb.Path(only_video).stem if only_video else None
    for actor in npy_root.iterdir():
        if not actor.is_dir():
            continue
        for seq in actor.iterdir():
            if only_seq_name and seq.name != only_seq_name:
                continue
            if seq.is_dir():
                audio_file = pipeline_paths["speech"] / actor.name / (seq.name + ".wav")
                process_one(
                    seq,
                    frame_root / actor.name / seq.name,
                    video_root / actor.name / (seq.name + ".mp4"),
                    video_3d_root / actor.name / (seq.name + ".mp4"),
                    audio_file,
                    expression_scale,
                    mouth_scale,
                    audio_mouth_strength,
                )
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render expression numpy outputs to image frames")
    add_data_root_arg(parser)
    parser.add_argument(
        "--expression-scale",
        type=float,
        default=float(os.environ.get("SFA_EXPRESSION_SCALE", DEFAULT_EXPRESSION_SCALE)),
        help="Multiplies predicted expression offsets during rendering only. Raw fake.npy is unchanged.",
    )
    parser.add_argument(
        "--mouth-scale",
        type=float,
        default=float(os.environ.get("SFA_MOUTH_SCALE", DEFAULT_MOUTH_SCALE)),
        help="Extra temporal amplification for likely mouth/lip channels during rendering only.",
    )
    parser.add_argument(
        "--audio-mouth-strength",
        type=float,
        default=float(os.environ.get("SFA_AUDIO_MOUTH_STRENGTH", DEFAULT_AUDIO_MOUTH_STRENGTH)),
        help="Adds speech-energy-driven motion to jaw/lip channels during rendering only.",
    )
    parser.add_argument("--only-video", default=None, help="Render only the output for this .mp4 filename")
    parser.add_argument("--only-media", default=None, help="Render only the output for this video or audio filename")
    parser.add_argument("--only-audio", default=None, help="Render only the output for this audio filename")
    args = parser.parse_args()
    process_all(
        args.data_root,
        args.expression_scale,
        args.mouth_scale,
        args.audio_mouth_strength,
        args.only_media or args.only_audio or args.only_video,
    )
    

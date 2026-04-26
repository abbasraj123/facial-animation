import pathlib as plb
import subprocess as sup
from project_paths import paths, parse_data_root, ensure_dirs

def extract_one_video(video_file, audio_file):
    command = 'ffmpeg -i "{}" -y -ab 160k -ac 2 -ar 44100 -vn "{}" -loglevel quiet'.format(video_file, audio_file)
    sup.call(command, shell=True)    
    
def make_dirs(video_root, audio_root):
    vr_dir = plb.Path(video_root)
    ar_dir = plb.Path(audio_root)
    ensure_dirs(ar_dir)
    for actor in vr_dir.iterdir():
        if actor.is_dir():
            ensure_dirs(ar_dir / actor.name)

def convert_all(video_root, audio_root):
    vr_dir = plb.Path(video_root)
    for actor in vr_dir.iterdir():
        if actor.is_file() and actor.suffix.lower() == ".mp4":
            audio_path = plb.Path(audio_root) / (actor.stem + ".wav")
            extract_one_video(str(actor), str(audio_path))
            continue
        if not actor.is_dir():
            continue
        for video_file in actor.iterdir():
            if video_file.suffix.lower() == '.mp4':
                video_path = str(video_file)
                audio_path = str(plb.Path(audio_root) / actor.name / (video_file.stem + ".wav"))
                extract_one_video(video_path, audio_path)

if __name__ == "__main__":
    data_root = parse_data_root("Extract wav audio files from videos")
    pipeline_paths = paths(data_root)
    video_root = pipeline_paths["video"]
    audio_root = pipeline_paths["speech"]
    make_dirs(video_root, audio_root)
    convert_all(video_root, audio_root)
    
    

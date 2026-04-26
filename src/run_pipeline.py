import argparse
import subprocess
import sys
import pathlib as plb

from project_paths import get_data_root, paths, ensure_dirs


STAGES = {
    "wav": "2.extract_wav.py",
    "features": "3.extract_feature.py",
    "spectrogram": "4.extract_spectrogram.py",
    "ctf": "6.create_spectrogram_CTF.py",
    "train": "7.train_end2end.py",
    "infer": "infer_pretrained.py",
    "eval": "9.eval_speech.py",
    "render": "shape_renderer.py",
}


def run_stage(script_dir, stage, data_root, extra_args):
    script = script_dir / STAGES[stage]
    command = [sys.executable, str(script), "--data-root", str(data_root)] + extra_args
    print("Running " + stage + ": " + " ".join(command))
    subprocess.check_call(command)


def main():
    parser = argparse.ArgumentParser(description="Run the speech-driven facial animation pipeline")
    parser.add_argument("--data-root", default=None, help="Single root folder for downloaded data and generated files")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=list(STAGES.keys()) + ["preprocess", "all"],
        default=["preprocess"],
        help="Stages to run. preprocess means wav features spectrogram.",
    )
    parser.add_argument("stage_args", nargs=argparse.REMAINDER, help="Extra args passed to each selected stage")
    args = parser.parse_args()

    data_root = get_data_root(args.data_root)
    pipeline_paths = paths(data_root)
    ensure_dirs(
        pipeline_paths["video"],
        pipeline_paths["speech"],
        pipeline_paths["features"],
        pipeline_paths["speech_dir"],
        pipeline_paths["exp_labels"],
        pipeline_paths["frontal_faces"],
        pipeline_paths["outputs"],
        pipeline_paths["models"],
    )

    selected = []
    for stage in args.stages:
        if stage == "preprocess":
            selected.extend(["wav", "features", "spectrogram"])
        elif stage == "all":
            selected.extend(["wav", "features", "spectrogram", "ctf", "train", "eval", "render"])
        else:
            selected.append(stage)

    script_dir = plb.Path(__file__).resolve().parent
    for stage in selected:
        run_stage(script_dir, stage, data_root, args.stage_args)


if __name__ == "__main__":
    main()

import argparse
import os
import pathlib as plb


DEFAULT_DATA_ROOT = plb.Path(__file__).resolve().parents[1] / "data"


def get_data_root(path=None):
    root = path or os.environ.get("SFA_DATA_ROOT") or DEFAULT_DATA_ROOT
    return plb.Path(root).expanduser().resolve()


def add_data_root_arg(parser):
    parser.add_argument(
        "--data-root",
        default=None,
        help="Root folder for all input data and generated pipeline files. Defaults to repo/data or SFA_DATA_ROOT.",
    )
    return parser


def parse_data_root(description):
    parser = argparse.ArgumentParser(description=description)
    add_data_root_arg(parser)
    args = parser.parse_args()
    return get_data_root(args.data_root)


def paths(data_root=None):
    root = get_data_root(data_root)
    return {
        "root": root,
        "video": root / "video",
        "speech": root / "speech",
        "features": root / "features",
        "speech_dir": root / "speech_dir",
        "exp_labels": root / "ExpLabels",
        "frontal_faces": root / "FrontalFaceData",
        "outputs": root / "outputs",
        "models": root / "models",
    }


def ensure_dirs(*dirs):
    for directory in dirs:
        plb.Path(directory).mkdir(parents=True, exist_ok=True)

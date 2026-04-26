# Running The Project

Create and activate the old Python environment first:

```powershell
conda create -n facial-animation python=3.6 -y
conda activate facial-animation
pip install numpy scipy opencv-python librosa pillow pyglet cntk
conda install -c conda-forge dlib -y
```

Install FFmpeg and confirm:

```powershell
ffmpeg -version
```

Put your downloaded data under `data/` using the layout in `data/README.md`.

Run preprocessing:

```powershell
python src/run_pipeline.py --data-root data --stages preprocess
```

Run a specific stage:

```powershell
python src/run_pipeline.py --data-root data --stages wav
python src/run_pipeline.py --data-root data --stages spectrogram
python src/run_pipeline.py --data-root data --stages infer
python src/run_pipeline.py --data-root data --stages eval
python src/run_pipeline.py --data-root data --stages render
```

For a single video without RAVDESS labels, use the pretrained inference stage:

```powershell
python src/run_pipeline.py --data-root data --stages infer
```

This accepts `.mp4` files placed either directly in `data/` or in `data/video/`.

Run everything:

```powershell
python src/run_pipeline.py --data-root data --stages all
```

Training expects CNTK CTF files and expression labels. If you only want to test the included sample output, open:

```text
sample/Example/video.mp4
```

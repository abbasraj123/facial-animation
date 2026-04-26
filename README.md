# Speech-Driven Facial Animation

This project generates a simple 3D facial animation from speech audio. It uses a pretrained CNTK model to map speech spectrogram features to 46 facial expression/blendshape coefficients, then renders those coefficients on a 3D face mesh.

The current repo supports inference without downloading the full RAVDESS dataset. You can place an `.mp4` file in `data/`, run the pretrained model, and export rendered videos with audio.

## What This Project Does

The pipeline has four main parts:

1. Extract audio from an input video.
2. Convert speech audio into dB spectrogram features.
3. Use the pretrained CNTK model to predict facial expression coefficients.
4. Render the predicted coefficients onto a 3D face and export videos.

The final output includes:

```text
3D-only rendered video
comparison video with original frame + neutral face + predicted face
raw model output as .npy files
rendered frame images
```

## Repository Structure

```text
Speech-driven-facial-animation/
  data/                         input videos and generated outputs
  Outputs/                      included pretrained CNTK model
  sample/                       original sample data and sample output
  shapes/                       3D face mesh assets
    baseshapes.npy
    triangles.npy
  src/                          pipeline scripts
    infer_pretrained.py         pretrained inference without RAVDESS labels
    shape_renderer.py           3D render and video export
    run_pipeline.py             stage runner
    project_paths.py            shared path configuration
  MajorProjectReport.pdf        original project report
  RUNNING.md                    short run notes
```

## Pretrained Model

The repo already contains a pretrained model:

```text
Outputs/model_audio2exp_2019-03-24-21-03/model_audio2exp_2019-03-24-21-03.dnn
```

You do not need to train the model again for normal inference. Training is only needed if you want to build a new model using a labeled dataset.

## Requirements

This is an old research codebase. Use Python 3.6 for best compatibility.

Recommended environment:

```powershell
conda create -n facial-animation python=3.6 -y
conda activate facial-animation
```

Install packages:

```powershell
pip install cntk librosa pillow opencv-python
pip install numpy==1.16.4 scipy==1.3.1 pyglet==1.5.27
conda install -c conda-forge dlib ffmpeg -y
```

Notes:

- CNTK 2.7 is old and may crash with newer NumPy/Scipy versions.
- `pyglet` must be 1.x because `pyglet 2.x` requires newer Python.
- FFmpeg is required for audio extraction and adding audio to the rendered videos.
- The CNTK GPU warning can be ignored when CPU inference completes successfully.

## Data Layout

For pretrained inference, the easiest option is to put videos directly inside:

```text
data/
```

Example:

```text
data/
  Recording 2026-04-26 151734.mp4
```

The pipeline will generate folders like:

```text
data/
  speech/                       extracted .wav files
  features/                     generated spectrogram CSV files
  outputs/
    npy_output/                 raw predicted coefficients
    rendered_frames/            rendered JPG frame sequences
    rendered_videos/            comparison videos
    rendered_videos_3d/         full 3D-only videos
```

## Run Inference And Render

From the project root:

```powershell
cd C:\Users\Rajpur\Desktop\proj\Speech-driven-facial-animation
conda activate facial-animation
```

Run all videos in `data/`:

```powershell
python src/run_pipeline.py --data-root data --stages infer render
```

Run one specific video only:

```powershell
python src\infer_pretrained.py --data-root data --only-video "Recording 2026-04-26 151734.mp4"
python src\shape_renderer.py --data-root data --only-video "Recording 2026-04-26 151734.mp4"
```

## Run With Desktop UI

You can also use the Tkinter UI:

```powershell
python src\ui_app.py
```

The UI lets you:

```text
select an .mp4 file
tune expression/lip movement values
run pretrained inference and rendering
watch progress and logs
open the generated comparison video
open the generated 3D-only video
```

If the selected video is outside `data/`, the UI copies it into `data/` before processing.

## Output Locations

Full 3D-only videos:

```text
data\outputs\rendered_videos_3d\default\
```

Comparison videos:

```text
data\outputs\rendered_videos\default\
```

Raw predicted expression coefficients:

```text
data\outputs\npy_output\default\<video-name>\fake.npy
```

Rendered frame images:

```text
data\outputs\rendered_frames\default\<video-name>\
```

For example:

```text
data\outputs\rendered_videos_3d\default\Recording 2026-04-26 151734.mp4
```

## Improving Lip Movement

The pretrained model often produces subtle facial motion. The renderer includes render-only enhancement controls. These do not modify the raw model output in `fake.npy`; they only affect the final video render.

Recommended stronger lip settings:

```powershell
$env:SFA_EXPRESSION_SCALE="7"
$env:SFA_MOUTH_SCALE="6"
$env:SFA_AUDIO_MOUTH_STRENGTH="1.2"
python src/run_pipeline.py --data-root data --stages render
```

Useful settings:

```text
SFA_EXPRESSION_SCALE         overall temporal expression amplification
SFA_MOUTH_SCALE              extra temporal boost for likely mouth/lip channels
SFA_AUDIO_MOUTH_STRENGTH     speech-energy-driven jaw/lip opening
```

If the mouth movement looks too exaggerated, reduce the values:

```powershell
$env:SFA_EXPRESSION_SCALE="5"
$env:SFA_MOUTH_SCALE="4"
$env:SFA_AUDIO_MOUTH_STRENGTH="0.7"
python src/run_pipeline.py --data-root data --stages render
```

## Pipeline Stages

The stage runner supports:

```powershell
python src/run_pipeline.py --data-root data --stages infer
python src/run_pipeline.py --data-root data --stages render
python src/run_pipeline.py --data-root data --stages infer render
```

Legacy/training-related stages also exist:

```text
wav
features
spectrogram
ctf
train
eval
```

For normal pretrained inference, use only:

```text
infer render
```

## Training

Training is not required for using the included pretrained model.

The original training flow expects RAVDESS/VIDTIMIT/SAVEE-style data with expression labels and CNTK CTF files. Without labeled expression data, you can run inference but you cannot train a new supervised model.

## Important Limitations

- This is an older research project, not a production lip-sync system.
- The pretrained model was trained on a specific data style, so results on arbitrary phone recordings may be limited.
- The current lip enhancement is render-side post-processing, not true phoneme/viseme modeling.
- Better lip sync would require a phoneme/viseme model or a newer approach such as Wav2Lip/SadTalker/FaceFormer-style pipelines.
- The renderer depends on OpenGL through `pyglet`; rendering can be sensitive to Python and graphics driver versions.

## Troubleshooting

If CNTK crashes during evaluation, make sure the environment uses compatible versions:

```powershell
pip install numpy==1.16.4 scipy==1.3.1
```

If rendering fails with `pyglet requires Python 3.8 or newer`, install the old compatible version:

```powershell
pip install pyglet==1.5.27
```

If rendered videos have no audio, confirm FFmpeg is installed inside the environment:

```powershell
conda activate facial-animation
ffmpeg -version
```

If a specific video fails at the final audio window, update to the current code where `src/extract_feature.py` zero-pads short final chunks.

## Quick Example

```powershell
cd C:\Users\Rajpur\Desktop\proj\Speech-driven-facial-animation
conda activate facial-animation
python src\infer_pretrained.py --data-root data --only-video "Recording 2026-04-26 151734.mp4"
python src\shape_renderer.py --data-root data --only-video "Recording 2026-04-26 151734.mp4"
```

Then open:

```text
data\outputs\rendered_videos_3d\default\Recording 2026-04-26 151734.mp4
```

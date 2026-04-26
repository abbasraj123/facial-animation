# Speech-Driven Facial Animation - Detailed Project Report

## 1. Project Title

**Speech-Driven Facial Animation Using Audio-Based Expression Prediction and 3D Blendshape Rendering**

## 2. Project Overview

This project generates a 3D facial animation from speech. The system accepts a video or audio file as input, extracts or converts the audio, converts the audio into spectrogram features, predicts facial expression coefficients using a pretrained deep learning model, and renders the predicted expression sequence on a 3D face mesh.

The core idea is that speech contains information about mouth movement, speaking rhythm, and emotional intensity. A neural network can learn a mapping from speech features to facial expression parameters. These parameters can then be used to animate a 3D face model.

The current implementation supports pretrained inference without requiring the full RAVDESS dataset. A user can select a video or audio file, run inference, and obtain rendered output videos.

## 3. Problem Statement

Manual facial animation is time-consuming and requires artistic effort. For applications such as virtual assistants, avatars, animation, telepresence, and talking agents, it is useful to automatically generate facial motion from speech.

The problem addressed by this project is:

```text
Given speech audio, predict facial expression/blendshape values that can animate a 3D face model.
```

The final system should:

- accept a normal video or audio input,
- extract speech audio,
- predict facial expression movement,
- render a 3D face animation,
- export videos that can be viewed by the user.

## 4. Objectives

The main objectives are:

- Build a speech-driven facial animation pipeline.
- Use a pretrained deep learning model for expression prediction.
- Render predicted expression coefficients on a 3D blendshape face.
- Provide a simple UI for selecting videos and generating outputs.
- Improve lip and jaw movement visibility using render-side enhancement.
- Make the project runnable without requiring full training data.

## 5. Existing System / Previous Model

The original codebase was a research-style implementation. It contained:

- numbered preprocessing scripts,
- hardcoded folder paths,
- a pretrained CNTK model,
- sample data,
- rendering utilities,
- training scripts for dataset-based experimentation.

The original workflow expected a strict dataset structure and several folders such as:

```text
../video
../speech
../speech_dir
../ExpLabels
../FrontalFaceData
```

The original implementation was difficult to run directly because:

- paths were hardcoded,
- the code expected RAVDESS-style dataset folders,
- inference expected labels even when running on a new video,
- rendering needed shape files in a specific relative location,
- generated videos did not include audio,
- the output was mostly raw `.npy` files or frame sequences,
- there was no polished UI.

## 6. Improvements Made In This Project

Several practical improvements were made to convert the research code into a runnable project.

### 6.1 Single Data Root

A new path management module was added:

```text
src/project_paths.py
```

This allows the project to use one consistent root folder:

```text
data/
```

Now all input videos, extracted audio, features, model outputs, rendered frames, and videos are organized under `data/`.

### 6.2 Inference Without Full RAVDESS Dataset

A new inference script was added:

```text
src/infer_pretrained.py
```

This allows the system to run on a normal `.mp4` video or standalone audio file without requiring RAVDESS expression labels.

The script:

- finds the input video/audio file,
- extracts audio,
- creates spectrogram features,
- loads the pretrained CNTK model,
- predicts 46 expression coefficients,
- saves outputs in `.npy` format.

### 6.3 Single-Video Processing

Support was added for processing only one selected video:

```powershell
python src\infer_pretrained.py --data-root data --only-video "video.mp4"
python src\shape_renderer.py --data-root data --only-video "video.mp4"
```

This prevents unnecessary reprocessing of all videos.

### 6.4 Shape Asset Loading Fixed

The renderer now correctly loads:

```text
shapes/baseshapes.npy
shapes/triangles.npy
```

Earlier, the code depended on relative paths that could fail depending on the working directory.

### 6.5 Rendered MP4 Output

The renderer was improved to export videos, not only image frames.

Output folders:

```text
data/outputs/rendered_videos/
data/outputs/rendered_videos_3d/
```

### 6.6 Audio Added To Rendered Videos

The original OpenCV video writer produced silent videos. FFmpeg muxing was added so the extracted audio is added back into the rendered videos.

### 6.7 Lip Movement Enhancement

The pretrained model sometimes produces subtle mouth movement. To improve visibility, render-side enhancements were added:

- global expression scaling,
- mouth/lip channel boosting,
- audio-energy-driven jaw/lip motion.

These settings improve the final render without modifying the raw model prediction.

Environment variables:

```text
SFA_EXPRESSION_SCALE
SFA_MOUTH_SCALE
SFA_AUDIO_MOUTH_STRENGTH
```

### 6.8 Final UI

A Tkinter-based desktop UI was added:

```text
src/ui_app.py
```

The UI allows users to:

- select a video file,
- tune render strength,
- choose presets,
- run inference and rendering,
- view logs,
- open generated output videos.

### 6.9 Compatibility Fixes

Several compatibility fixes were made:

- Python 3.6-compatible subprocess handling,
- CNTK DLL path handling,
- NumPy/Scipy version recommendations,
- `pyglet==1.5.27` requirement,
- zero-padding for short final audio windows.

## 7. System Architecture

The system pipeline is:

```text
Input MP4 Video
      |
      v
Audio Extraction using FFmpeg
      |
      v
Speech Spectrogram Feature Extraction
      |
      v
Pretrained CNTK Model
      |
      v
46 Expression / Blendshape Coefficients
      |
      v
3D Face Renderer using baseshapes.npy and triangles.npy
      |
      v
Rendered Frame Sequence
      |
      v
MP4 Video Export with Audio
```

## 8. Module Description

### 8.1 `infer_pretrained.py`

Responsible for pretrained inference.

Main tasks:

- locate input videos,
- extract audio,
- compute spectrograms,
- normalize feature size to model input size,
- load CNTK model,
- predict facial expression coefficients,
- save `fake.npy`, `real.npy`, and `img.npy`.

### 8.2 `shape_renderer.py`

Responsible for rendering predicted expressions.

Main tasks:

- load predicted expression arrays,
- load 3D face mesh assets,
- apply render-side expression/lip enhancement,
- generate rendered frames,
- export comparison video,
- export 3D-only video,
- add audio to rendered MP4s.

### 8.3 `ShapeUtils.py`

Contains 3D shape and OpenGL rendering utilities.

Main tasks:

- load baseshapes,
- load triangle mesh,
- calculate facial mesh from expression coefficients,
- render the mesh,
- capture rendered image.

### 8.4 `project_paths.py`

Central path configuration module.

It defines standard folders:

```text
data/video
data/speech
data/features
data/outputs
data/models
data/ExpLabels
```

### 8.5 `ui_app.py`

Desktop interface for non-technical users.

Features:

- video/audio selection,
- render tuning sliders,
- processing progress,
- log panel,
- output file buttons.

## 9. Model Details

The project uses a pretrained CNTK model:

```text
model_audio2exp_2019-03-24-21-03.dnn
```

Model input:

```text
Sequence[Tensor[1, 128, 32]]
```

This means each audio frame is represented as a `128 x 32` spectrogram feature.

Model output:

```text
Sequence[Tensor[46]]
```

This means the model predicts 46 facial expression/blendshape coefficients per time step.

## 10. Input And Output

### Input

The current system takes:

```text
.mp4 video file or audio file
```

For video input, the video is used for:

- extracting audio,
- showing original frames in the comparison output.

The deep learning model itself uses only:

```text
audio-derived spectrogram features
```

For audio-only input, the system skips original video frame extraction and renders the 3D animation directly from the audio.

### Output

The system generates:

```text
fake.npy       predicted expression coefficients
real.npy       neutral placeholder for no-label inference
img.npy        extracted input frames
rendered JPG frames
comparison MP4 video
3D-only MP4 video
```

Important output folders:

```text
data/outputs/rendered_videos/default/
data/outputs/rendered_videos_3d/default/
```

## 11. User Interface

The UI is designed for easier demonstration.

It provides:

- file browser,
- selected video details,
- render tuning sliders,
- presets such as Natural, Balanced, and Strong Lips,
- progress bar,
- processing log,
- output file open buttons.

Run UI:

```powershell
python src\ui_app.py
```

## 12. Technologies Used

```text
Python 3.6
CNTK 2.7
NumPy
SciPy
Librosa
OpenCV
Pyglet / OpenGL
FFmpeg
Tkinter
```

## 13. Why CNTK?

The original pretrained model was created using Microsoft CNTK. To use the existing `.dnn` model directly, CNTK is required.

However, CNTK is now outdated, which creates compatibility challenges. This is why Python 3.6 and older library versions are used.

## 14. Why Use Spectrograms?

Raw audio waveform is difficult for a model to interpret directly. A spectrogram represents audio in terms of frequency and time. Speech sounds and mouth movements are strongly related to frequency-time patterns.

The model uses dB spectrogram features of size:

```text
128 x 32
```

These features capture short-time speech information useful for predicting facial expressions.

## 15. Why Use Blendshapes?

Blendshapes are a common method for facial animation. A neutral face mesh is combined with multiple expression basis shapes. Each coefficient controls the strength of one expression.

Advantages:

- efficient rendering,
- easy expression control,
- common in animation and game pipelines,
- works well with neural network output.

## 16. Result Explanation

The generated comparison video has three sections:

```text
Left   - original input video frame
Middle - neutral/reference 3D face
Right  - predicted animated 3D face
```

The 3D-only video shows only the predicted rendered face and is better for viewing the final animation.

## 17. Limitations

Current limitations:

- The pretrained model is old.
- The model was trained on a specific dataset style.
- Arbitrary phone recordings may not produce perfect lip sync.
- Lip enhancement is render-side post-processing, not true phoneme detection.
- CNTK dependency makes setup old and fragile.
- The 3D face model is generic, not the same person as the input video.

## 18. Future Improvements

Possible improvements:

- Add audio-only input support.
- Add phoneme or viseme detection for better lip sync.
- Replace CNTK with PyTorch or TensorFlow.
- Train a newer model on larger audiovisual datasets.
- Add real-time preview in the UI.
- Add a 3D viewer inside the UI.
- Improve mouth shape mapping using known viseme classes.
- Support multiple face meshes.
- Export FBX/GLTF animation data.
- Add emotion control sliders.

## 19. Difference From Original Project

| Area | Original Codebase | Improved Project |
|---|---|---|
| Input | Dataset-style folders | Direct `.mp4` video support |
| Paths | Hardcoded `../` paths | Central `data/` root |
| Inference | Expected labels/dataset | Pretrained no-label inference |
| Output | `.npy` and frames | MP4 videos with audio |
| Rendering | Fragile shape paths | Robust shape asset loading |
| Lip movement | Subtle raw prediction | Render-side mouth/audio boost |
| UI | Notebook/basic scripts | Polished Tkinter UI |
| Single file | Not supported cleanly | `--only-video` support |
| Usability | Research-code workflow | Demonstration-ready workflow |

## 20. Commands For Demo

Run UI:

```powershell
python src\ui_app.py
```

Run one video from terminal:

```powershell
python src\infer_pretrained.py --data-root data --only-video "Recording 2026-04-26 151734.mp4"
python src\shape_renderer.py --data-root data --only-video "Recording 2026-04-26 151734.mp4"
```

Run all videos:

```powershell
python src\run_pipeline.py --data-root data --stages infer render
```

Tune stronger lip movement:

```powershell
$env:SFA_EXPRESSION_SCALE="7"
$env:SFA_MOUTH_SCALE="6"
$env:SFA_AUDIO_MOUTH_STRENGTH="1.2"
python src\run_pipeline.py --data-root data --stages render
```

## 21. Likely Viva / Teacher Questions And Answers

### Q1. What is the main goal of this project?

The goal is to generate 3D facial animation automatically from speech audio. The system predicts facial expression coefficients from audio and renders those coefficients on a 3D face mesh.

### Q2. What is the input to the model?

The model input is not the full video. The model receives audio-derived spectrogram features of shape `1 x 128 x 32` over time.

### Q3. Why does the application accept a video if the model uses audio?

The video is used as a convenient container. The system extracts audio from it for prediction and also extracts frames for the comparison output. The model prediction itself is based on audio.

### Q4. Can the system work with audio only?

Yes. Audio-only support was added. The system accepts formats such as `.wav`, `.mp3`, `.m4a`, `.aac`, `.flac`, and `.ogg`, converts them to WAV when needed, predicts expression coefficients, and exports a 3D face animation with the audio attached.

### Q5. What is the model output?

The model outputs 46 expression or blendshape coefficients per time step. These values control the 3D face animation.

### Q5. What is a blendshape?

A blendshape is a facial deformation basis. A face mesh can be animated by combining a neutral mesh with weighted expression shapes. The model predicts these weights.

### Q6. What is CNTK?

CNTK is Microsoft Cognitive Toolkit, an older deep learning framework. The pretrained `.dnn` model in this project was trained using CNTK, so CNTK is required to load and run it.

### Q7. Why are older Python versions required?

CNTK 2.7 is outdated and works best with older Python and NumPy versions. Python 3.6 with compatible NumPy/Scipy versions provides better stability.

### Q8. Why do we use spectrogram features?

Spectrograms represent audio as frequency-time information. Speech sounds are easier for a neural network to analyze in spectrogram form compared to raw waveform.

### Q9. What are `baseshapes.npy` and `triangles.npy`?

`baseshapes.npy` stores the neutral face and expression deformation bases. `triangles.npy` stores mesh connectivity used to render the 3D face.

### Q10. Why was lip movement initially weak?

The pretrained model produced subtle expression changes for arbitrary videos. Also, the generic blendshape mesh does not always strongly show small coefficient changes. Render-side enhancement was added to make lip movement more visible.

### Q11. Is the lip enhancement changing the model?

No. It only affects rendering. The raw prediction file `fake.npy` remains unchanged.

### Q12. What improvement was made to lip movement?

Three improvements were added:

- expression temporal scaling,
- mouth-specific channel boost,
- audio-energy-driven jaw/lip motion.

### Q13. Does this system generate the same person’s face?

No. It animates a generic 3D face model. The input video is used for speech audio and comparison frames, not for reconstructing the person’s actual face.

### Q14. What is the difference between comparison video and 3D-only video?

The comparison video shows original frame, neutral 3D face, and predicted 3D face side by side. The 3D-only video shows only the predicted rendered face.

### Q15. Can the system run without RAVDESS?

Yes, for inference. The pretrained model can run on a normal `.mp4` video. RAVDESS-style data is only required for training a new model.

### Q16. Can we train this model again?

Yes, but training requires labeled expression data and CNTK-compatible CTF files. Without expression labels, supervised training is not possible.

### Q17. What were the major engineering improvements?

The major improvements were:

- central data-root path handling,
- inference-only mode,
- single-video processing,
- MP4 output generation,
- audio muxing,
- render tuning,
- UI development,
- compatibility fixes.

### Q18. What are the limitations of the current system?

The main limitations are:

- old CNTK dependency,
- generic 3D face,
- limited lip-sync accuracy,
- render-side enhancement instead of true viseme modeling,
- dependency on old Python environment.

### Q19. How can this project be improved in the future?

Future work can include:

- replacing CNTK with PyTorch,
- using phoneme-to-viseme mapping,
- training on modern audiovisual datasets,
- adding real-time preview,
- supporting personalized 3D faces,
- improving emotion control.

### Q20. Why is this useful?

It can be used in virtual avatars, animation tools, online teaching assistants, gaming characters, talking agents, and human-computer interaction systems.

## 22. Conclusion

This project demonstrates an end-to-end speech-driven facial animation pipeline. It takes a normal video file, extracts speech audio, predicts expression coefficients using a pretrained model, and renders an animated 3D face. The project was improved from a research-code workflow into a usable application with single-video inference, organized data paths, audio-enabled video export, stronger lip motion, and a Tkinter UI.

The system shows how deep learning and 3D graphics can be combined to automate facial animation from speech.

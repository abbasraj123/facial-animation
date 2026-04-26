# Data Folder

Put downloaded/input files here. All updated scripts now use this folder by default.

Default root:

```text
Speech-driven-facial-animation/data
```

You can also use another folder with:

```powershell
python src/run_pipeline.py --data-root D:\my-animation-data
```

Expected layout:

```text
data/
  video/                 input .mp4 files, either flat or grouped by actor
    Actor_01/
      01-01-01-01-01-01-01.mp4
  speech/                generated .wav files
  features/              generated mfcc/log_mel/chroma/dbspectrogram files
    Actor_01/
      01-01-01-01-01-01-01/
        dbspectrogram.csv
  speech_dir/            optional old CNTK training/evaluation spectrogram layout
    RAVDESS/
    RAVDESS_feat/
  ExpLabels/             .npy expression labels
    RAVDESS/
      Actor_01/
        01-01-01-01-01-01-01.npy
  FrontalFaceData/       extracted/cropped frame folders for evaluation rendering
    Actor_01/
      01-01-01-01-01-01-01/
        frame0.jpg
  models/                trained/pretrained .dnn files
    model_audio2exp_2019-03-24-21-03.dnn
  outputs/               generated numpy outputs and rendered frames
```

The repository also contains a fallback pretrained model in `Outputs/model_audio2exp_2019-03-24-21-03/`.

For pretrained inference only, a single `.mp4` may also be placed directly in `data/`.

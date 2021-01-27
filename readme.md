# DeepCube: Transcribing Rubik’s Cube Moves with Action Recognition

## video

video presentation at 
https://youtu.be/RxsI5vvTb7g 

## saved models

download the saved model weights from 

https://drive.google.com/drive/folders/1UZbwK2z4bBqkwBPiEsAhEZBkhvKNRlfE?usp=sharing

extract `saved_models/` folder to the repo root

## environment

```
conda create -y -n cube python=3.7
conda activate cube
conda install -y matplotlib
pip install opencv-python flask flask-cors torchvision tqdm
cd src/ctcdecode
git submodule update --init --recursive
pip install .
```

## data

`data/video/` contains raw recordings in mp4 format in the following specs:

- video specs todo

`data/label/` contains json files of the labels corresponding to each video that share the same filename before the file extension, in the following format:

```json
{
  "num_frames": 263,
  "moves": {
    "35": "U",
    "64": "L",
    "107": "R'",
    "131": "x",
    "167": "F",
    "207": "B'",
    "242": "U"
  }
}
```
`data/label_cropped` contains the temporally localized annotation for all moves, in the following format:
```json
{
    "label_fname + end_frame + label_i": {
        "begin_frame": 2,
        "end_frame": 10,
        "duration": 8,
        "vspec_id": "label_fname",   
        "action_label": "U"
    }
}
```
```label_fname``` corresponds to the original json file name in ```data/label_aligned/```

You may run ```python reprocess_labels.py``` to generate the preprocessed video segment in .pkl for video classification.

## visualizer

First, start the server for the data files, `python src/playback_server/app.py &`.

Then, start the client for the visualizer, `cd src/playback_tool && npm start`.

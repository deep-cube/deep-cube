import torch
import os
import json
import numpy as np
import data_utils

ACCEPTED_VIDEO_EXTENSION = ['mp4']
# number of frames before first label that can be sampled
# number of frames after last label that can be sampled
NUM_VALID_FRAMES_BEYOND_LABEL = 10


def walk_video_filenames(video_path):
    return [
        f for f in sorted(os.listdir(video_path))
        if os.path.isfile(os.path.join(video_path, f))
        and f.split(os.extsep)[-1].lower() in ACCEPTED_VIDEO_EXTENSION
    ]


def get_json_filenames(video_filenames):
    return [
        (''.join(f.split('.')[:-1]) + '.json')
        for f in video_filenames
    ]

def get_filenames_wo_ext(raw_filenames):
    return [
        (''.join(f.split('.')[:-1]))
        for f in raw_filenames
    ]

def load_json(file_path):
    with open(file_path) as f:
        label_dict = json.load(f)
    
    return label_dict

def load_json_labels(label_path, json_filenames):
    label_dicts = [None] * len(json_filenames)
    for i in range(len(json_filenames)):
        with open(os.path.join(label_path, json_filenames[i])) as f:
            label_dict = json.load(f)
            int_converted_moves = {
                int(idx): label_dict['moves'][idx]
                for idx in label_dict['moves']
            }
            label_dict['moves'] = int_converted_moves
            label_dicts[i] = label_dict
    return label_dicts

def load_json_labels_with_fname(label_path, json_filenames):
    label_dicts = {}
    filenames = get_filenames_wo_ext(json_filenames)

    for i in range(len(json_filenames)):
        with open(os.path.join(label_path, json_filenames[i])) as f:
            label_dict = json.load(f)
            int_converted_moves = {
                int(idx): label_dict['moves'][idx]
                for idx in label_dict['moves']
            }
            label_dict['moves'] = int_converted_moves
            label_dicts[filenames[i]] = label_dict
    return label_dicts


def populate_segmentable_ranges(label_dicts, video_specs, clip_length):
    for i, label_dict in enumerate(label_dicts):
        smallest_frame_idx = min(i for i in label_dict['moves'])
        largest_frame_idx = max(i for i in label_dict['moves'])

        # check no json label start with frame 0,
        # json file with move at frame 0 signals the json file is not aligned yet
        assert smallest_frame_idx > 0, \
            f'{video_specs[i]["filename"]} has a move at frame 0, looks like it is not aligned yet'

        min_seg_frame = max(smallest_frame_idx -
                            NUM_VALID_FRAMES_BEYOND_LABEL, 0)
        max_seg_frame = min(
            largest_frame_idx + NUM_VALID_FRAMES_BEYOND_LABEL + 1,
            video_specs[i]['num_frames'] - clip_length - 1)

        video_specs[i]['min_seg_frame'] = min_seg_frame
        video_specs[i]['max_seg_frame'] = max_seg_frame


def filter_labels(segmented_label_dict, video_filenames, min_accepted_frames, exclude_filenames=[]):
    """
    To filter segmented labels
    """
    filtered_label_dict = {}
    vfile_names = [x for x in video_filenames if x not in exclude_filenames]
    vfile_names = get_filenames_wo_ext(vfile_names)

    for key_name in segmented_label_dict:
        for vname in vfile_names:
            # make sure only include the desired file name, 
            if vname in key_name:
                elem = segmented_label_dict[key_name]
                # make sure at least x frames
                if elem['duration'] >= min_accepted_frames:
                    filtered_label_dict[key_name] = elem

    return filtered_label_dict



                



        



















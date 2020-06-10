"""
Task:
1. Convert all videos in to small clips in tensor with name:
videoName_frameId_actionName.pkl

2. Convert all labels to json with structure:
{
    "label_fname + end_frame + label_i": {
        "begin_frame": 2,
        "end_frame": 10,
        "duration": 8,
        "vspec_id": "label_fname",   # the name of the video_specs in video_spec_dict
        "action_label": "U"
        ""
    },
    " ... ": {
        ...
    }
}
"""

import json
import os
import data_loader
import data_loader_util
import data_utils
from reprocess_labels_utils import *


VIDEO_PATH = data_loader.DEFAULT_VIDEO_PATH
LABEL_PATH = data_loader.DEFAULT_LABEL_PATH

TEST_FILE = "russell_stable3"


def generate_clips_label(dict_label_list, video_specs_dict, save_to_file = False):
    """
    label_data = {
        "label_fname + end_frame + label_i": {
            "begin_frame": 2,
            "end_frame": 10,
            "duration": 8,
            "vspec_id": "label_fname",   # the name of the video_specs in video_spec_dict
            "action_label": "U"
            ""
        }
    }
    """
    label_data = {}

    offset = 5
    min_frames = 3
    max_frames = 50

    for label_fname in dict_label_list:

        label_pairs = dict_label_list[label_fname]
        offset_begin_frame = max(label_pairs[0][0] - 10, 0) + offset
        video_max_frames = video_specs_dict[label_fname]['num_frames']

        for i, (end_frame, label_i) in enumerate(label_pairs):
            
            offset_end_frame = end_frame + offset

            # remove bad labels
            if offset_begin_frame >= video_max_frames or offset_end_frame >= video_max_frames:
                break

            # adjust offset of end from for (super fast next move)
            if i + 1 < len(label_pairs):
                offset_next_end_frame = label_pairs[i + 1][0] + offset
                next_duration =  abs(offset_next_end_frame - offset_end_frame)
                
                if next_duration < min_frames:
                    offset_end_frame = end_frame + next_duration // 2
            
            # check duration
            duration = abs(offset_end_frame - offset_begin_frame)
            
            if duration < min_frames:
                offset_begin_frame = offset_end_frame
                continue
            elif duration > max_frames:
                offset_begin_frame = offset_end_frame - max_frames
            
            data_dict ={}
            data_dict['begin_frame'] = offset_begin_frame
            data_dict['end_frame'] = offset_end_frame
            data_dict['duration'] = offset_end_frame - offset_begin_frame
            data_dict['vspec_id'] = label_fname
            data_dict['action_label'] = label_i
            key_name = label_fname + "_" + str(end_frame) + "_" + label_i
            label_data[key_name] = data_dict

            assert data_dict['duration'] >= min_frames, "duration is wrong"
            assert offset_begin_frame < video_max_frames, "begin frame is beyond num_frames {} in {}".format(video_max_frames, key_name)
            assert offset_end_frame < video_max_frames, "end frame is beyond num_frames {} in {}".format(video_max_frames, key_name)

            offset_begin_frame = offset_end_frame

    if save_to_file:
        save_path = '../data/label_cropped/annotation.json'
        with open(save_path, 'w') as fp:
            json.dump(label_data, fp)
            print("Generated annotation file to {}".format(save_path))

    return label_data


def generate_video_clips(video_specs_dict, label_data, start_point = 0, num_files = 10):

    print("There are total {} data labels".format(len(label_data)))

    count = 0 
    for fname in label_data:
        if count < start_point:
            count += 1
            continue 

        data_dict = label_data[fname]
        data_utils.save_cropped_video(video_specs_dict[data_dict['vspec_id']], 
            data_dict['begin_frame'], data_dict['duration'], output_name = fname)

        count += 1
        if count > start_point + num_files:
            break

def generate_pickle_files(video_specs_dict, label_data, test_generate=False):

    count = 0
    if test_generate:
        print(" Test pickle file generation...")
    else:
        print(" There are total {} files to process".format(len(label_data)))

    for fname in label_data:

        if test_generate and TEST_FILE not in fname:
            continue

        data_dict = label_data[fname]
        data_utils.save_cropped_pickle(video_specs_dict[data_dict['vspec_id']], 
            data_dict['begin_frame'], data_dict['duration'], output_name = fname)

        if count % 1000 == 0 or count + 1 == len(label_data):
            print("Processed {} files".format(count))
        count += 1

    if not test_generate:
        print("Generated total {} data labels".format(len(label_data)))


def load_data(video_path, label_path):

    video_filenames = data_loader_util.walk_video_filenames(video_path)
    json_filenames = data_loader_util.get_json_filenames(video_filenames)

    # load json labels
    label_dicts = data_loader_util.load_json_labels_with_fname(
        label_path, json_filenames)

    video_specs = [
        data_loader.get_video_specs(os.path.join(video_path, f))
        for f in video_filenames
    ]

    num_videos = len(video_specs)

    return video_filenames, json_filenames, label_dicts, video_specs, num_videos


def trim_video(video_path, label_path):

    (video_filenames, json_filenames, label_dicts, 
        video_specs, num_videos) = load_data(video_path, label_path)

    dict_label_list = convert_label_dict_to_sorted_list(label_dicts)
    video_specs_dict = make_video_spec_dict(video_specs)

    # generate trimmed label.
    label_data = generate_clips_label(dict_label_list, video_specs_dict, save_to_file=True)
    # generate trimmed video.
    generate_pickle_files(video_specs_dict, label_data, test_generate=True)


def main_worker():

    trim_video(VIDEO_PATH, LABEL_PATH)


    # (video_filenames, json_filenames, label_dicts, 
    #     video_specs, num_videos) = load_data(VIDEO_PATH, LABEL_PATH)

    # dict_label_list = convert_label_dict_to_sorted_list(label_dicts)

    # video_specs_dict = make_video_spec_dict(video_specs)

    # get_label_statistics(dict_label_list, show_label_distribution=True, show_duration=True, print_stats=True)

    # label_data = generate_clips_label(dict_label_list, video_specs_dict, save_to_file=True)

    # generate_video_clips(video_specs_dict, label_data, start_point = 0, num_files = 11)
    # generate_video_clips(video_specs_dict, label_data, start_point = 800, num_files = 11)
    # generate_video_clips(video_specs_dict, label_data, start_point = 2600, num_files = 11)
    # generate_video_clips(video_specs_dict, label_data, start_point = 4707, num_files = 11)
    # generate_video_clips(video_specs_dict, label_data, start_point = 7777, num_files = 11)
    # generate_video_clips(video_specs_dict, label_data, start_point = 11865, num_files = 11)
    # generate_video_clips(video_specs_dict, label_data, start_point = 14345, num_files = 11)
    # generate_video_clips(video_specs_dict, label_data, start_point = 16999, num_files = 11)
    # generate_video_clips(video_specs_dict, label_data, start_point = 18999, num_files = 11)


if __name__ == "__main__":

    main_worker()





































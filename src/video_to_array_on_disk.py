import pickle
import os
import sys
import cv2
import data_loader_util
import data_utils

if __name__ == "__main__":
    srcdir = sys.argv[1]
    dstdir = sys.argv[2]

    video_filenames = data_loader_util.walk_video_filenames(srcdir)

    for video_filename in video_filenames:
        video_path = os.path.join(srcdir, video_filename)
        video_spec = data_utils.get_video_specs(video_path)
        video_array = data_utils.video_to_array(
            video_spec,
            0,
            video_spec['num_frames'] - 1
        )

        with open(os.path.join(dstdir, f'{video_filename}.pkl'), 'wb') as f:
            pickle.dump(video_array, f)



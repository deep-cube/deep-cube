import cv2
import os
import numpy as np
import pickle

NUM_CHANNEL = 3
DEFAULT_OUTPUT_DIR = "../data/cropped_test"
DEFAULT_OUTPUT_NAME = "crop_test.mp4"
DEFAULT_PICKLE_DIR = "../data/preprocessed"

DEFAULT_HEIGHT = 68
DEFAULT_WIDTH = 90


def num_valid_segments(lower_bound, upper_bound, clip_length):
    """calculate the number of valid video clips in the video

    Args:
    - lower_bound (int): denotes the earliest frame in video that can be segmented
    - upper_bound (int): denotes the latest frame + 1 in video that can be segmented
    - clip_length (int): length of each clip

    Returns:
    - int: number of valid clips in th evideo
    """
    return upper_bound - clip_length - lower_bound


def get_video_specs(path):
    """return the specs of a video file

    Args:
    - path (string): path to the video

    Returns:
    - dict: containing video spec
    """
    filename = path.split(os.sep)[-1]
    cap = cv2.VideoCapture(path)
    return {
        'filename': filename,
        'filepath': path,
        'num_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": int(cap.get(cv2.CAP_PROP_FPS)),
    }


def video_to_array(video_specs, start_frame, clip_length):
    """load part of a video, return as array

    Args:
    - video_specs (dict): containing video spec, return value of get_video_spec
    - start_frame (int): frame index of beginning of clip
    - clip_length (int): length of clip to return 

    Returns:
    - np.array: of (L, H, W, C), frame data
    """
    num_frames = video_specs['num_frames']
    height = video_specs['height']
    width = video_specs['width']
    assert start_frame + clip_length < num_frames, \
        f'frame {start_frame + clip_length} beyond total of {num_frames} frames'

    buffer = np.empty(
        [clip_length, height, width, NUM_CHANNEL],
        np.dtype('uint8')
    )

    cap = cv2.VideoCapture(video_specs['filepath'])
    # skip to
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_idx in range(clip_length):
        ret, buffer[frame_idx] = cap.read()
    cap.release()

    return buffer


def save_cropped_pickle(video_specs, start_frame, clip_length, output_dir=DEFAULT_PICKLE_DIR, output_name="test.pkl"):
    
    video_array = video_to_array(video_specs, start_frame, clip_length)

    with open(os.path.join(output_dir, f'{output_name}.pkl'), 'wb') as f:
        pickle.dump(video_array, f)



def save_cropped_video(video_specs, start_frame, clip_length, output_dir=DEFAULT_OUTPUT_DIR, output_name=DEFAULT_OUTPUT_NAME):
    """load part of a video, save to output path

    Args:
    - video_specs (dict): containing video spec, return value of get_video_spec
    - start_frame (int): frame index of beginning of clip
    - clip_length (int): length of clip to save

    Returns:
    - None
    """
    num_frames = video_specs['num_frames']
    height = video_specs['height']
    width = video_specs['width']
    fps = video_specs['fps']
    assert start_frame + clip_length < num_frames, \
        f'frame {start_frame + clip_length} beyond total of {num_frames} frames'

    # Get the input video and skip to clip
    cap = cv2.VideoCapture(video_specs['filepath'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Get the output video config
    out = cv2.VideoWriter(os.path.join(output_dir, f'{output_name}.mp4'),cv2.VideoWriter_fourcc('F','M','P','4'), fps, (width,height))
    
    for frame_idx in range(clip_length):

        ret, frame = cap.read()

        if ret == True:
            out.write(frame)
        else:
            break

    out.release()
    cv2.destroyAllWindows()

def array_to_video_view(x, y=None):
    L, C, H, W = x.shape
    for l in range(L):
        frame = x[l].transpose(1, 2, 0).astype(np.uint8)
        print(f'frame={l} label={y[l] if y is not None else "?"}')
        cv2.imshow(f'frame{l}', frame)

        if cv2.waitKey() == ord('q'):
            break

        cv2.destroyAllWindows()
    cv2.destroyAllWindows()

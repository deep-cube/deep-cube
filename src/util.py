# np array with shape (frames, height, width, channels)
import math
import numpy as np
from matplotlib import pyplot as plt
import torch
from data_def import CLASS_STRS

def show_video(video, start=0, end=None, layout='square', label=None):
    if end is None:
        end = min(25 + start, video.shape[0])

    #plt.figure(figsize=(20, 10))
    _, vH, vW = video[0].shape
    margin = int(0.1 * vH)

    r_shift = 18
    c_shift= 1

    if layout=='square':
      span = math.ceil(math.sqrt(end - start))
      H, W = (vH + margin) * span, (vW + margin) * span
    elif layout=='row':
      span = end - start
      H, W = vH + margin, (vW + margin) * span

    grid = np.full((H, W, 3), 255)
    for idx in range(start, end):
        r, c = (idx - start) // span, (idx - start) % span
        frame = video[idx]
        frame = np.moveaxis(frame, 0, -1).copy()
        if frame.shape[2] == 3: # if not grayscale, switch R and B
            ch2 = frame[:, :, 2].copy()
            frame[:, :, 2] = frame[:, :, 0]
            frame[:, :, 0] = ch2
        grid[r * (vH + margin) : (r + 1) * vH + r * margin, \
             c * (vW + margin) : (c + 1) * vW + c * margin] = frame

        if label is not None:
          txt = CLASS_STRS[label[idx]]
          if txt != "_":
            plt.text(c * (vW + margin) + c_shift, r * (vH + margin) + r_shift, \
            txt, color="white",  fontsize=16, fontweight='bold')
    plt.axis("off")
    plt.imshow(grid.astype('uint8'))


def bgr_2_rgb(frame):
    ch = frame[:, :, 2].copy()
    frame[:, :, 2] = frame[:, :, 0]
    frame[:, :, 0] = ch
    return frame

def show_video_heatmap(video, start=0, end=None, layout='square', label=None, margin = None):
    if end is None:
        end = min(25 + start, video.shape[0])

    #plt.figure(figsize=(20, 10))
    r_shift = 16
    c_shift= 1
    vH, vW = video[0].shape
    if margin is None:
      margin = int(0.1 * vH)
    if layout=='square':
      span = math.ceil(math.sqrt(end - start))
      H, W = (vH + margin) * span, (vW + margin) * span
    elif layout=='row':
      span = end - start
      H, W = vH + margin, (vW + margin) * span

    grid = np.full((H, W), 255)
    for idx in range(start, end):
        r, c = (idx - start) // span, (idx - start) % span
        frame = video[idx]
        grid[r * (vH + margin) : (r + 1) * vH + r * margin, \
             c * (vW + margin) : (c + 1) * vW + c * margin] = frame
          
        if label is not None:
          txt = CLASS_STRS[label[idx]]
          if txt != "_":
            plt.text(c * (vW + margin) + c_shift, r * (vH + margin) + r_shift, \
            txt, color="white",  fontsize=16, fontweight='bold')
    plt.axis("off")
    plt.imshow(grid.astype('uint8'), cmap=plt.cm.hot)


import data_def
from data_loader import VideoDataset

def load_datasets(clip_length):
  L = clip_length
  train_files = [
            'kevin_random_moves_quick.mp4',
            'kevin_random_moves_quick_2.mp4',
            'kevin_random_moves_quick_3.mp4',
            'kevin_random_moves_quick_4.mp4',
            'kevin_random_moves_slow.mp4',
            'kevin_rotate_1.mp4',
            'kevin_simple_shuffle_1.mp4',
            'kevin_single_moves_2.mp4',
            'kevin_single_solve_1.mp4',
            'kevin_solve_play_1.mp4',
            'kevin_solve_play_10.mp4',
            'kevin_solve_play_11.mp4',
            'kevin_solve_play_12.mp4',
            'kevin_solve_play_13.mp4',
            'kevin_solve_play_2.mp4',
            'kevin_solve_play_3.mp4',
            'kevin_solve_play_6.mp4',
            'kevin_solve_play_7.mp4',
            'kevin_solve_play_8.mp4',
            'kevin_solve_play_9.mp4',
            'zhouheng_cfop_solve.mp4',
            'zhouheng_oll_algorithm.mp4',
            'zhouheng_pll_algorithm_fast.mp4',
            'zhouheng_rotation.mp4',
            'zhouheng_scramble_01.mp4',
            'zhouheng_scramble_03.mp4',
            'zhouheng_weird_turns.mp4',
        ]
  dev_files = [
              'kevin_single_moves_1.mp4',
              'kevin_solve_play_5.mp4',
              'zhouheng_scramble_02.mp4',
          ]
  train_dataset = VideoDataset(
          '../data/video_small',
          '../data/label_aligned',
          clip_length=L,
          verbose_init=True,
          label_conversion_func=data_def.sparse_to_expanded_array,
          video_filenames=train_files
      )
  dev_dataset = VideoDataset(
          '../data/video_small',
          '../data/label_aligned',
          clip_length=L,
          verbose_init=True,
          label_conversion_func=data_def.sparse_to_expanded_array,
          video_filenames=dev_files
      )

  return train_dataset, dev_dataset
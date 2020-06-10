import torch
import argparse
import os
import json
import data_def
import numpy as np
from pprint import pprint
from tqdm import tqdm
from data_utils import *
import data_loader_util
import reprocess_labels_utils
import spatial_transforms
from PIL import Image

DEFAULT_VIDEO_PATH = '../data/video_small'
DEFAULT_LABEL_PATH = '../data/label_aligned'
DEFAULT_SEGMENTED_LABEL_PATH = '../data/label_cropped/annotation.json'

DEFAULT_CLIP_LENGTH = 200  # frames
DATALOADER_NUM_WORKERS = 1

TRAIN_FILENAMES_MINI = [
    'russell_stable3.mp4',
]

TRAIN_FILENAMES = [
    'kevin_random_moves_quick.mp4',
    'kevin_random_moves_quick_2.mp4',
    'kevin_random_moves_quick_3.mp4',
    'kevin_random_moves_quick_4.mp4',
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
    'kevin_solve_play_7.mp4',
    'kevin_solve_play_8.mp4',
    'kevin_solve_play_9.mp4',
    'russell_scramble0.mp4',
    'russell_scramble1.mp4',
    'russell_scramble3.mp4',
    'russell_scramble4.mp4',
    'russell_scramble5.mp4',
    'russell_scramble7.mp4',
    'russell_stable0.mp4',
    'russell_stable1.mp4',
    'russell_stable2.mp4',
    'russell_stable3.mp4',
    'zhouheng_cfop_solve.mp4',
    'zhouheng_long_solve_1.mp4',
    'zhouheng_long_solve_2.mp4',
    'zhouheng_long_solve_3.mp4',
    'zhouheng_oll_algorithm.mp4',
    'zhouheng_pll_algorithm_fast.mp4',
    'zhouheng_rotation.mp4',
    'zhouheng_scramble_01.mp4',
    'zhouheng_scramble_03.mp4',
    'zhouheng_weird_turns.mp4',
]
DEV_FILENAMES = [
    'kevin_single_moves_1.mp4',
    'kevin_solve_play_6.mp4',
    'zhouheng_scramble_02.mp4',
    'russell_scramble2.mp4',
]
TEST_FILENAMES = [
    'zhouheng_long_solve_5.mp4',
    'kevin_solve_play_5.mp4',
    'zhouheng_long_solve_4.mp4',
    'russell_scramble6.mp4',
]

EXCLUDE_FILENAMES = [
    # 'kevin_solve_play_1.mp4',
    # 'kevin_solve_play_10.mp4',
    # 'kevin_solve_play_11.mp4',
    # 'kevin_solve_play_12.mp4',
    # 'kevin_solve_play_13.mp4',
    # 'kevin_solve_play_2.mp4',
    # 'kevin_solve_play_3.mp4',
    # 'kevin_solve_play_7.mp4',
    # 'kevin_solve_play_8.mp4',
    # 'kevin_solve_play_9.mp4',
    # 'kevin_solve_play_6.mp4',
    # 'kevin_solve_play_5.mp4',
]


def get_train_data(B, L, verbose_init=True, shorten_factor=1, spatial_augment=True):
    train_dataset = VideoDataset(
        DEFAULT_VIDEO_PATH,
        DEFAULT_LABEL_PATH,
        clip_length=L,
        verbose_init=verbose_init,
        video_filenames=TRAIN_FILENAMES,
        shorten_factor=shorten_factor,
        spatial_augment=spatial_augment,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=B, num_workers=DATALOADER_NUM_WORKERS,
    )
    return train_dataset, train_loader


def get_dev_data(B, L, verbose_init=True):
    dev_dataset = VideoDatasetNoSample(
        DEFAULT_VIDEO_PATH,
        DEFAULT_LABEL_PATH,
        clip_length=L,
        verbose_init=verbose_init,
        video_filenames=DEV_FILENAMES
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=B, num_workers=DATALOADER_NUM_WORKERS,
    )
    return dev_dataset, dev_loader


def get_eval_dev_data(video_filenames, overlap_length=15):
    dev_dataset = VideoDatasetOverlapped(
        DEFAULT_VIDEO_PATH,
        DEFAULT_LABEL_PATH,
        video_filenames=video_filenames,
        overlap_length=overlap_length
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=1, num_workers=DATALOADER_NUM_WORKERS,
    )
    return dev_dataset, dev_loader


def get_segmented_train_data(B, L, spatial_transform=None, temporal_shift=False, verbose_init=True):
    print("\nGetting TRAIN data...")
    train_dataset = VideoDatasetSegmented(
        DEFAULT_VIDEO_PATH,
        DEFAULT_SEGMENTED_LABEL_PATH,
        spatial_transform=spatial_transform,
        temporal_shift=temporal_shift,
        verbose_init=verbose_init,
        min_accepted_frames=L,
        video_filenames=TRAIN_FILENAMES
#         video_filenames=TRAIN_FILENAMES_MINI
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=B, num_workers=DATALOADER_NUM_WORKERS,
    )
    return train_dataset, train_loader


def get_segmented_dev_data(B, L, verbose_init=True):
    print("\nGetting DEV data...")
    dev_dataset = VideoDatasetSegmented(
        DEFAULT_VIDEO_PATH,
        DEFAULT_SEGMENTED_LABEL_PATH,
        verbose_init=verbose_init,
        min_accepted_frames=L,
        video_filenames=DEV_FILENAMES
#         video_filenames=TRAIN_FILENAMES_MINI
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_dataset, batch_size=B, num_workers=DATALOADER_NUM_WORKERS,
    )
    return dev_dataset, dev_loader


def get_segmented_eval_data(L, verbose_init=True):
    print("\nGetting TEST data...")
    test_dataset = VideoDatasetSegmented(
        DEFAULT_VIDEO_PATH,
        DEFAULT_SEGMENTED_LABEL_PATH,
        verbose_init=verbose_init,
        min_accepted_frames=L,
        video_filenames=TEST_FILENAMES
        # video_filenames=TRAIN_FILENAMES_MINI
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=5, num_workers=DATALOADER_NUM_WORKERS,
    )
    return test_dataset, test_loader


class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        video_path,
        label_path,
        clip_length=DEFAULT_CLIP_LENGTH,
        verbose_init=False,
        video_filenames=None,
        shorten_factor=1,
        spatial_augment=True,
    ):
        self.clip_length = clip_length
        self.spatial_augment = spatial_augment

        if video_filenames is None:
            video_filenames = data_loader_util.walk_video_filenames(video_path)

        json_filenames = data_loader_util.get_json_filenames(video_filenames)
        for f in json_filenames:
            assert os.path.exists(
                os.path.join(label_path, f)
            ), f'Found video, but [{f}] does not exist.'

        # load json labels
        self.label_dicts = data_loader_util.load_json_labels(
            label_path, json_filenames)

        # load video specs
        self.video_specs = [
            get_video_specs(os.path.join(video_path, f))
            for f in video_filenames
        ]
        self.num_videos = len(self.video_specs)

        # check that all videos have same height and width
        assert len(set([spec['height'] for spec in self.video_specs])) == 1, \
            'videos are of different height'
        assert len(set([spec['width'] for spec in self.video_specs])) == 1, \
            'videos are of different width'

        data_loader_util.populate_segmentable_ranges(
            self.label_dicts, self.video_specs, self.clip_length
        )

        # check that sampleable segment frames are at least clip_length in each video
        for spec in self.video_specs:
            assert spec['max_seg_frame'] - spec['min_seg_frame'] >= clip_length,\
                f'{spec["filename"]} does not have at least {clip_length} frames within sample-able range'

        # when sampling, randomly choose a video proportional to its (num_frames - clip_length + 1)
        # such that any valid sampleable segments in the dataset has an equal chance of selected
        total_num_valid_segments = [
            num_valid_segments(
                spec['min_seg_frame'], spec['max_seg_frame'], clip_length
            )
            for spec in self.video_specs
        ]
        self.random_video_p = np.array(
            total_num_valid_segments) / np.sum(total_num_valid_segments)

        # since we're randomly sampling clip_length frames per sample, we say that the length of
        # this dataset is however many expected sample needed to span number of total frames
        self.len = int(np.ceil(sum(total_num_valid_segments) /
                               clip_length / shorten_factor))

        self.random_state = None

        if verbose_init:
            print('VideoDataset __init__')
            print('clip_length:', clip_length)
            print(
                'frame size:', self.video_specs[0]['height'], 'x', self.video_specs[0]['width']
            )
            print('video files:')
            pprint(video_filenames)
            # print('json files:', json_filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.random_state is None:
            self.random_state = np.random.RandomState()

        # discard idx, sample frames and return
        video_idx = self._choose_video()
        seg_start, seg_end = self._choose_segment_range(video_idx)

        x = video_to_array(
            self.video_specs[video_idx],
            start_frame=seg_start,
            clip_length=seg_end - seg_start,
        )  # (clip_length, h, w, c)
        if self.spatial_augment:
            x = spatial_transforms.default_spatial_augment(x)

        y_expanded, _ = data_def.sparse_to_expanded_array(
            self.label_dicts[video_idx]['moves'],
            seg_start, seg_end,
            self.clip_length
        )  # (clip_length,)

        y_collapsed, y_collapsed_l = data_def.sparse_to_collapsed_array(
            self.label_dicts[video_idx]['moves'],
            seg_start, seg_end,
            self.clip_length
        )

        return (
            x.transpose(0, 3, 1, 2),  # (clip_length, c, h, w)
            y_expanded,
            y_collapsed,
            y_collapsed_l
        )

    def _choose_video(self):
        ''' return random index of video, chose proportional to segmentable length '''
        return self.random_state.choice(self.num_videos, p=self.random_video_p)

    def _choose_segment_range(self, video_idx):
        start_offset = self.random_state.choice(
            num_valid_segments(
                self.video_specs[video_idx]['min_seg_frame'],
                self.video_specs[video_idx]['max_seg_frame'],
                self.clip_length
            )
        )
        start = self.video_specs[video_idx]['min_seg_frame'] + start_offset
        end = start + self.clip_length
        assert end <= self.video_specs[video_idx]['num_frames']  # sanity
        return (start, end)

    def frame_size(self):
        return self.video_specs[0]['height'], self.video_specs[0]['width']

    def num_class(self):
        return data_def.NUM_CLASS

    def get_label_counts(self):
        d = {}
        total_moves = 0
        for label_dict in self.label_dicts:
            for frame in label_dict['moves']:
                move = label_dict['moves'][frame]
                if move not in d:
                    d[move] = 0
                d[move] += 1
                total_moves += 1

        total_frames = sum(spec['num_frames'] for spec in self.video_specs)
        empty_frames = total_frames - total_moves
        d[data_def.NO_MOVE_STR] = empty_frames

        return d


class VideoDatasetNoSample(torch.utils.data.Dataset):
    def __init__(
        self,
        video_path,
        label_path,
        clip_length=DEFAULT_CLIP_LENGTH,
        verbose_init=False,
        video_filenames=None,
    ):
        self.clip_length = clip_length

        if video_filenames is None:
            video_filenames = data_loader_util.walk_video_filenames(video_path)

        # load json labels
        json_filenames = data_loader_util.get_json_filenames(video_filenames)
        self.label_dicts = data_loader_util.load_json_labels(
            label_path, json_filenames)

        # load video specs
        self.video_specs = [
            get_video_specs(os.path.join(video_path, f))
            for f in video_filenames
        ]
        self.num_videos = len(self.video_specs)

        data_loader_util.populate_segmentable_ranges(
            self.label_dicts, self.video_specs, self.clip_length
        )

        self.clip_configs = []
        for video_index in range(len(self.video_specs)):
            spec = self.video_specs[video_index]
            begin_frame = 0
            while True:
                end_frame = begin_frame + self.clip_length
                if end_frame >= spec['max_seg_frame']:
                    break
                self.clip_configs.append((video_index, begin_frame, end_frame))
                begin_frame = end_frame

        # print(self.clip_configs)
        self.len = len(self.clip_configs)

        if verbose_init:
            print('VideoDatasetNoSample __init__')
            print('clip_length:', clip_length)
            print(
                'frame size:', self.video_specs[0]['height'], 'x', self.video_specs[0]['width']
            )
            print('video files:')
            pprint(video_filenames)
            # print('json files:', json_filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        video_idx, seg_start, seg_end = self.clip_configs[idx]

        # discard idx, sample frames and return
        x = video_to_array(
            self.video_specs[video_idx],
            start_frame=seg_start,
            clip_length=seg_end - seg_start,
        )  # (clip_length, h, w, c)

        y_expanded, _ = data_def.sparse_to_expanded_array(
            self.label_dicts[video_idx]['moves'],
            seg_start, seg_end,
            self.clip_length
        )  # (clip_length,)

        y_collapsed, y_collapsed_l = data_def.sparse_to_collapsed_array(
            self.label_dicts[video_idx]['moves'],
            seg_start, seg_end,
            self.clip_length
        )

        return (
            x.transpose(0, 3, 1, 2),  # (clip_length, c, h, w)
            y_expanded,
            y_collapsed,
            y_collapsed_l
        )


class VideoDatasetOverlapped(torch.utils.data.Dataset):
    def __init__(
        self,
        video_path,
        label_path,
        video_filenames,
        clip_length=2000,
        overlap_length=15,
    ):
        self.clip_length = clip_length
        self.overlap_length = overlap_length

        # load json labels
        json_filenames = data_loader_util.get_json_filenames(video_filenames)
        self.label_dicts = data_loader_util.load_json_labels(
            label_path, json_filenames)

        # load video specs
        self.video_specs = [
            get_video_specs(os.path.join(video_path, f))
            for f in video_filenames
        ]
        self.num_videos = len(self.video_specs)
        data_loader_util.populate_segmentable_ranges(
            self.label_dicts, self.video_specs, 1
        )

        self.clip_configs = []
        for video_index in range(len(self.video_specs)):
            spec = self.video_specs[video_index]
            begin_frame = spec['min_seg_frame']
            while True:
                end_frame = begin_frame + self.clip_length
                is_last_clip = False
                if end_frame >= spec['max_seg_frame']:
                    end_frame = spec['max_seg_frame']
                    is_last_clip = True
                self.clip_configs.append((
                    video_index,
                    max(begin_frame-self.overlap_length, 0),
                    min(end_frame+self.overlap_length, spec['num_frames']-1)
                ))
                begin_frame = end_frame
                if is_last_clip:
                    break

        self.len = len(self.clip_configs)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        video_idx, seg_start, seg_end = self.clip_configs[idx]

        # discard idx, sample frames and return
        x = video_to_array(
            self.video_specs[video_idx],
            start_frame=seg_start,
            clip_length=seg_end - seg_start,
        )  # (clip_length, h, w, c)

        y_expanded, _ = data_def.sparse_to_expanded_array(
            self.label_dicts[video_idx]['moves'],
            seg_start, seg_end,
            seg_end - seg_start
        )  # (clip_length,)

        y_collapsed, y_collapsed_l = data_def.sparse_to_collapsed_array(
            self.label_dicts[video_idx]['moves'],
            seg_start, seg_end,
            seg_end - seg_start
        )

        return (
            x.transpose(0, 3, 1, 2),  # (clip_length, c, h, w)
            y_expanded,
            y_collapsed,
            y_collapsed_l
        )


class VideoDatasetSegmented(torch.utils.data.Dataset):
    def __init__(
        self,
        video_path,
        label_path,
        spatial_transform=None,
        temporal_shift=False,
        min_accepted_frames=5,
        verbose_init=False,
        video_filenames=None,
    ):
        if video_filenames is None:
            video_filenames = data_loader_util.walk_video_filenames(video_path)

        label_dict_full = data_loader_util.load_json(label_path)
        self.label_dict = data_loader_util.filter_labels(
            label_dict_full, video_filenames, min_accepted_frames, exclude_filenames=EXCLUDE_FILENAMES)

        self.fname_list = list(self.label_dict.keys())

        test_file_path = os.path.join(
            DEFAULT_PICKLE_DIR, f'{self.fname_list[0]}.pkl')
        # (L, H, W, C)
        self.file_specs = np.load(test_file_path, allow_pickle=True).shape

        self.len = len(self.fname_list)

        self.random_state = None
        self.clip_length = min_accepted_frames

        self.spatial_transform = spatial_transform
        self.temporal_shift = temporal_shift

        self.prev_x = None
        self.next_x = None

        if verbose_init:
            print('min_accepted_frames:', min_accepted_frames)
            print(
                'frame size:', self.file_specs
            )
            print('num_clips:', self.len)
            
            print('apply_spatial_transform:', self.spatial_transform is not None)
            print('apply_temporal_shift:', self.temporal_shift)
#             print('video files:')
#             pprint(video_filenames)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if self.random_state is None:
            self.random_state = np.random.RandomState()

        preprocessed_array_path = os.path.join(
            DEFAULT_PICKLE_DIR, f'{self.fname_list[idx]}.pkl')

        x = np.load(preprocessed_array_path, allow_pickle=True)
        x = self._choose_frames(x)
        x = self._temporal_shift(x, idx)
        x = self._spatial_transform(x)

        class_label = self.label_dict[self.fname_list[idx]]['action_label']
        y = data_def.CLASS_STR_TO_IDX[class_label]

        return (
            x.transpose(3, 0, 1, 2),  # (channel, clip_length, h, w)
            y,
        )

    def _choose_frames(self, x):
        ''' x should be [length, h, w, c] '''
        selected_idx = self.random_state.choice(
            range(len(x)), self.clip_length, replace=False)
        selected_idx.sort()
        return x[selected_idx]

    def _spatial_transform(self, x):
        """
        @param
        x (numpy): shape of (clip_length, h, w, channel)

        @return
        (numpy): same shape of x

        """
        if self.spatial_transform == None:
            return x

        self.spatial_transform.randomize_parameters()
        PIL_img_arr = [Image.fromarray(
            img.astype('uint8'), 'RGB') for img in x]
        output = [np.array(self.spatial_transform(img)) for img in PIL_img_arr]

        x_out = np.array(output)
        assert x_out.shape == x.shape, "WRONG: spatial transform changed the shape, \
        or perhaps x is not the default shape ({}, {})".format(data_utils.DEFAULT_WIDTH, data_utils.DEFAULT_HEIGHT)

        return x_out

    def _temporal_shift(self, x, idx):

        if self.temporal_shift == False:
            return x

        # skip the first and last video
        if idx <= 0 or idx >= self.len - 1:
            return x
        
        shift_amount = random_state.choice([-3,-2,-1,0,1,2,3])

        prev_x_path = os.path.join(
            DEFAULT_PICKLE_DIR, f'{self.fname_list[idx-1]}.pkl')

        next_x_path = os.path.join(
            DEFAULT_PICKLE_DIR, f'{self.fname_list[idx+1]}.pkl')

        # [TODO: sample from prev/next video]
        return x
        

    def frame_size(self):
        # Height, Width
        return self.file_specs[1], self.file_specs[2]

    def num_class(self):
        return data_def.NUM_CLASS

    def get_label_counts(self):
        d = {}
        total_moves = 0
        total_frames = 0

        for key in self.label_dict:
            move = label_dict[key]['action_label']
            total_frames += label_dict[key]['duration']
            if move not in d:
                d[move] = 0
            d[move] += 1
            total_moves += 1

        # there is no empty class in this case
        return d


if __name__ == "__main__":
    import data_utils

    dataset = VideoDataset(
        '../data/video_small',
        '../data/label_aligned',
        clip_length=100,
        verbose_init=True,
        spatial_augment=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    total_len = 0
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=20,
        num_workers=1,  # memory overhead is yuge, use only 1 worker to small buffer
    )

    for epoch in tqdm(range(2), desc='epoch'):

        for i, data in enumerate(tqdm(loader, desc='batch')):
            (x, y_expanded,
             y_collapsed,
             y_collapsed_l) = data
    #        x = x.to(device, torch.float32)

            B, L, C, H, W = x.shape
            total_len += B * L

        print(total_len)

    # # sanity check video_to_array
    # video_path = '../data/video/alpha1.mp4'
    # clip_array = video_to_array(
    #     get_video_specs(video_path),
    #     20,
    #     20,
    # )
    # print(clip_array)
    # print((clip_array[0] == clip_array[-1]).all())
    # for i in range(len(clip_array)):
    #     cv2.namedWindow(f'frame{i}')
    #     cv2.imshow(f'frame{i}', clip_array[i])
    #     cv2.waitKey(0)

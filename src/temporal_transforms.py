import random
import math
import numpy as np
import data_utils



def get_transform_func(clip_length, random_sample=True, random_shift=True):

	pass


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, frame_indices):
        for i, t in enumerate(self.transforms):
            if isinstance(frame_indices[0], list):
                next_transforms = Compose(self.transforms[i:])
                dst_frame_indices = [
                    next_transforms(clip_frame_indices)
                    for clip_frame_indices in frame_indices
                ]

                return dst_frame_indices
            else:
                frame_indices = t(frame_indices)
        return frame_indices



class RandomSample(object):
	
	def __init__(self, clip_length):
		self.clip_length = clip_length

	def __call__(self, x, random_state):

		# select L frames w/o replacement
		selected_idx = random_state.choice(
            range(len(x)), self.clip_length, replace=False)
        
        # sort the frames
        selected_idx.sort()

        return x[selected_idx]



class RandomShift(object):
	def __init__(self, clip_length, shift_range=3):
		self.clip_length = clip_length
		self.shfit_range = shift_range

	def __call__(self, prev_x, x, next_x, random_state):
		pass

		




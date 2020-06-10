import random
import numpy as np
import data_utils

from torchvision.transforms import transforms
from torchvision.transforms import functional as F
from PIL import Image


TEST_IMAGE_PATH = '../data/preprocessed/russell_stable3_3367_F.pkl'


def default_spatial_augment(
    x,
    random_crop=True,
    color_jitter=True,
):
    transform = get_transform_func(
        random_crop=random_crop, color_jitter=color_jitter)
    transform.randomize_parameters()
    PIL_img_arr = [Image.fromarray(img.astype('uint8'), 'RGB') for img in x]
    output = [np.array(transform(img)) for img in PIL_img_arr]
    x_out = np.array(output)

    return x_out


def get_transform_func(
        image_height=data_utils.DEFAULT_HEIGHT,
        image_width=data_utils.DEFAULT_WIDTH,
        random_crop=True, color_jitter=True):
    """
    return the spatial_transform function, enable the features you want.
    """
    spatial_transform = []

    if random_crop:
        end_scale = 0.75
        num_scales = 5
        scales = [1.0]
        scale_step = end_scale ** (1 / (num_scales - 1))
        for _ in range(1, num_scales):
            scales.append(scales[-1] * scale_step)
        spatial_transform.append(MultiScaleCornerCrop(
            image_height, image_width, scales))

    if color_jitter:
        spatial_transform.append(ColorJitter())

    spatial_transform = Compose(spatial_transform)

    return spatial_transform


class Compose(transforms.Compose):
    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(transforms.ToTensor):

    def randomize_parameters(self):
        pass


class Normalize(transforms.Normalize):

    def randomize_parameters(self):
        pass


class CornerCrop(object):

    def __init__(self,
                 height_size,
                 width_size,
                 crop_position=None,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br']):
        self.height_size = height_size
        self.width_size = width_size
        self.crop_position = crop_position
        self.crop_positions = crop_positions

        if crop_position is None:
            self.randomize = True
        else:
            self.randomize = False
        self.randomize_parameters()

    def __call__(self, img):
        """
        img is PIL Image with RGB 
        """
        image_width = img.size[0]
        image_height = img.size[1]

        h, w = (self.height_size, self.width_size)
        if self.crop_position == 'c':
            i = int(round((image_height - h) / 2.))
            j = int(round((image_width - w) / 2.))
        elif self.crop_position == 'tl':
            i = 0
            j = 0
        elif self.crop_position == 'tr':
            i = 0
            j = image_width - w
        elif self.crop_position == 'bl':
            i = image_height - h
            j = 0
        elif self.crop_position == 'br':
            i = image_height - h
            j = image_width - w

        img = F.crop(img, i, j, h, w)

        return img

    def randomize_parameters(self):
        if self.randomize:
            self.crop_position = self.crop_positions[random.randint(
                0,
                len(self.crop_positions) - 1)]

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, crop_position={1}, randomize={2})'.format(
            self.size, self.crop_position, self.randomize)


class MultiScaleCornerCrop(object):

    def __init__(self,
                 height,
                 width,
                 scales,
                 crop_positions=['c', 'tl', 'tr', 'bl', 'br'],
                 interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.scales = scales
        self.interpolation = interpolation
        self.crop_positions = crop_positions

        self.randomize_parameters()

    def __call__(self, img):
        height_size = int(img.size[1] * self.scale)
        width_size = int(img.size[0] * self.scale)
        self.corner_crop.height_size = height_size
        self.corner_crop.width_size = width_size

        img = self.corner_crop(img)
        # size – The requested size in pixels, as a 2-tuple: (width, height).
        return img.resize((self.width, self.height), self.interpolation)

    def randomize_parameters(self):
        self.scale = self.scales[random.randint(0, len(self.scales) - 1)]
        crop_position = self.crop_positions[random.randint(
            0,
            len(self.crop_positions) - 1)]

        self.corner_crop = CornerCrop(None, crop_position)

    def __repr__(self):
        return self.__class__.__name__ + '(h,w={0},{1}, scales={2}, interpolation={3})'.format(
            self.heigth, self.width, self.scales, self.interpolation)


class ColorJitter(transforms.ColorJitter):

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.randomize_parameters()

    def __call__(self, img):
        if self.randomize:
            self.transform = self.get_params(self.brightness, self.contrast,
                                             self.saturation, self.hue)
            # don't randomize for the rest of the clips
            self.randomize = False

        return self.transform(img)

    def randomize_parameters(self):
        self.randomize = True


def validate_img_arr(numpy_imgs):

    for img in numpy_imgs:
        validate_img(img)


def validate_img(numpy_img):
    assert np.amin(numpy_img, axis=None) >= 0 and np.amax(
        numpy_img, axis=None) <= 255, "WRONG: image has max {} and min {}".format(np.amax(numpy_img), np.amin(numpy_img))
    assert len(numpy_img.shape) == 3 and numpy_img.shape[0] == 3, "WRONG: the img has shape {}".format(
        numpy_img.shape)
    assert (numpy_img.dtype) == 'uint8', "WRONG: image element type is not integer of uint8, current type is {}".format(
        numpy_img.dtype)


if __name__ == '__main__':
    numpy_imgs = np.load(TEST_IMAGE_PATH, allow_pickle=True)
    numpy_imgs = numpy_imgs.transpose(0, 3, 1, 2)
    clip_length = 7

    random_state = np.random.RandomState()
    selected_idx = random_state.choice(
        range(len(numpy_imgs)), clip_length, replace=False)
    selected_idx.sort()

    # x = numpy_imgs[selected_idx]
    # PIL_img_arr = [Image.fromarray(img.astype('uint8'), 'RGB') for img in x]
    # output = [np.array(img) for img in PIL_img_arr]

    # x_out = np.array(output)
    # print( np.where(np.equal(x, x_out) == False) )

    img_arr = []
    for i, img in enumerate(numpy_imgs[selected_idx]):
        validate_img(img)
        # PIL accepts img of (H,W,C)
        pil_img = Image.fromarray(
            img.transpose(1, 2, 0).astype('uint8'), 'RGB')
        print("PIL image size:", pil_img.size)
        img_arr.append(pil_img)
        pil_img.save("../data/{}.png".format(i))

    end_scale = 0.75
    num_scales = 5
    scales = [1.0]
    scale_step = end_scale ** (1 / (num_scales - 1))
    for _ in range(1, num_scales):
        scales.append(scales[-1] * scale_step)

    image_width = pil_img.size[0]
    image_height = pil_img.size[1]

    spatial_transform = get_transform_func()

    for i, pil_img in enumerate(img_arr):
        pil_img = spatial_transform(pil_img)
        pil_img.save("../data/{}_transformed_nojitter.png".format(i))

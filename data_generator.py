from keras.preprocessing.image import *
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from PIL import Image
import numpy as np
import os
import cv2


def random_crop(x, random_crop_size, data_format, sync_seed=None, **kwargs):
    "channel last"
    # pylint: disable=no-member
    np.random.seed(sync_seed)
    h, w = x.shape[0], x.shape[1]
    rangeh = (h - random_crop_size[0]) // 2
    rangew = (w - random_crop_size[1]) // 2
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)

    h_start, h_end = offseth, offseth + random_crop_size[0]
    w_start, w_end = offsetw, offsetw + random_crop_size[1]
    return x[h_start:h_end, w_start:w_end, :]


def pair_random_crop(x, y, random_crop_size, data_format, sync_seed=None, **kwargs):
    # pylint: disable=no-member
    np.random.seed(sync_seed)
    # channel last
    h, w = x.shape[0], x.shape[1]
    rangeh = (h - random_crop_size[0]) // 2
    rangew = (w - random_crop_size[1]) // 2
    offseth = 0 if rangeh == 0 else np.random.randint(rangeh)
    offsetw = 0 if rangew == 0 else np.random.randint(rangew)

    h_start, h_end = offseth, offseth + random_crop_size[0]
    w_start, w_end = offsetw, offsetw + random_crop_size[1]
    # channel last
    return x[h_start:h_end, w_start:w_end, :], y[h_start:h_end, w_start:w_end, :]


class SegDirectoryIterator(Iterator):
    '''
    Users need to ensure that all files exist.
    Label images should be png images where pixel values represents class number.

    find images -name *.jpg > images.txt
    find labels -name *.png > labels.txt

    for a file name 2011_002920.jpg, each row should contain 2011_002920

    file_path: location of train.txt, or val.txt in PASCAL VOC2012 format,
        listing image file path components without extension
    data_dir: location of image files referred to by file in file_path
    label_dir: location of label files
    data_suffix: image file extension, such as `.jpg` or `.png`
    label_suffix: label file suffix, such as `.png`, or `.npy`
    loss_shape: shape to use when applying loss function to the label data
    '''

    def __init__(self, file_path, seg_data_generator,
                 data_dir, data_suffix,
                 label_dir, label_suffix, classes, ignore_label=255,
                 label_cval=255, pad_size=None,
                 target_size=None, color_mode='rgb',
                 class_mode='sparse',
                 batch_size=1, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg',
                 loss_shape=None):
        data_format = K.image_data_format()
        self.file_path = file_path
        self.data_dir = data_dir
        self.data_suffix = data_suffix
        self.label_suffix = label_suffix
        self.label_dir = label_dir
        self.classes = classes
        self.seg_data_generator = seg_data_generator
        self.target_size = tuple(target_size)
        self.ignore_label = ignore_label
        self.label_cval = label_cval
        self.pad_size = pad_size
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.data_format = data_format
        self.nb_label_ch = 1
        self.loss_shape = loss_shape

        if (self.label_suffix == '.npy') or (self.label_suffix == 'npy'):
            self.label_file_format = 'npy'
        else:
            self.label_file_format = 'img'
        if target_size:
            if self.color_mode == 'rgb':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = self.target_size + (1,)

            self.label_shape = self.target_size + (self.nb_label_ch,)
        elif batch_size != 1:
            raise ValueError(
                'Batch size must be 1 when target image size is undetermined')
        else:
            self.image_shape = None
            self.label_shape = None
        if class_mode not in {'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of '
                             '"sparse", or None.')
        self.class_mode = class_mode
        if save_to_dir:
            self.palette = None
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        white_list_formats = {'png', 'jpg', 'jpeg', 'bmp', 'npy'}

        # build lists for data files and label files
        self.data_files = []
        self.label_files = []
        fp = open(file_path)
        lines = fp.readlines()
        fp.close()
        self.nb_sample = len(lines)
        for line in lines:
            line = line.strip('\n')
            self.data_files.append(line + data_suffix)
            self.label_files.append(line + label_suffix)
        super(SegDirectoryIterator, self).__init__(
            self.nb_sample, batch_size, shuffle, seed)

    def next(self):
        with self.lock:
            index_array, current_index, current_batch_size = next(
                self.index_generator)

        # The transformation of images is not under thread lock so it can be
        # done in parallel
        if self.target_size:
            batch_x = np.zeros((current_batch_size,) + self.image_shape)
            if self.loss_shape is None and self.label_file_format is 'img':
                batch_y = np.zeros((current_batch_size,) + self.label_shape,
                                   dtype=int)
            elif self.loss_shape is None:
                batch_y = np.zeros((current_batch_size,) + self.label_shape)
            else:
                batch_y = np.zeros((current_batch_size,) + self.loss_shape,
                                   dtype=np.uint8)
        grayscale = self.color_mode == 'grayscale'
        # build batch of image data and labels
        for i, j in enumerate(index_array):
            data_file = self.data_files[j]
            label_file = self.label_files[j]
            img_file_format = 'img'
            img = load_img(os.path.join(self.data_dir, data_file),
                           grayscale=grayscale, target_size=None)
            label_filepath = os.path.join(self.label_dir, label_file)

            if self.label_file_format == 'npy':
                y = np.load(label_filepath)
            else:
                label = Image.open(label_filepath)
                if self.save_to_dir and self.palette is None:
                    self.palette = label.palette

            # do padding
            if self.target_size:
                x = img_to_array(img, data_format=self.data_format)
                if self.label_file_format is not 'npy':
                    y = img_to_array(
                        label, data_format=self.data_format).astype(int)
                img_w, img_h = img.size
                if self.pad_size:
                    pad_w = max(self.pad_size[1] - img_w, 0)
                    pad_h = max(self.pad_size[0] - img_h, 0)
                else:
                    pad_w = max(self.target_size[1] - img_w, 0)
                    pad_h = max(self.target_size[0] - img_h, 0)
                x = np.lib.pad(x, ((pad_h / 2, pad_h - pad_h / 2), (pad_w / 2, pad_w - pad_w / 2), (0, 0)), 'constant', constant_values=0.)
                y = np.lib.pad(y, ((pad_h / 2, pad_h - pad_h / 2), (pad_w / 2, pad_w - pad_w / 2), (0, 0)), 'constant', constant_values=self.label_cval)

            if self.target_size is None:
                batch_x = np.zeros((current_batch_size,) + x.shape)
                if self.loss_shape is not None:
                    batch_y = np.zeros((current_batch_size,) + self.loss_shape)
                else:
                    batch_y = np.zeros((current_batch_size,) + y.shape)

            x, y = self.seg_data_generator.random_transform(x, y, self.target_size)

            if self.ignore_label:
                y[np.where(y == self.ignore_label)] = self.classes

            if self.loss_shape is not None:
                y = np.reshape(y, self.loss_shape)

            batch_x[i] = x
            batch_y[i] = y
        # optionally save augmented images to disk for debugging purposes
        if self.save_to_dir:
            # pylint: disable=no-member
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.data_format, scale=True)
                label = batch_y[i][:, :, 0].astype('uint8')
                label[np.where(label == self.classes)] = self.ignore_label
                label = Image.fromarray(label, mode='P')
                label.palette = self.palette
                fname = '{prefix}_{index}_{hash}'.format(prefix=self.save_prefix,
                                                         index=current_index + i,
                                                         hash=np.random.randint(1e4))
                img.save(os.path.join(self.save_to_dir, 'img_' +
                                      fname + '.{format}'.format(format=self.save_format)))
                label.save(os.path.join(self.save_to_dir,
                                        'label_' + fname + '.png'))
        # return
        batch_x = preprocess_input(batch_x)
        if self.class_mode == 'sparse':
            return batch_x, batch_y
        else:
            return batch_x


class SegDataGenerator(object):

    def __init__(self,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 zoom_maintain_shape=True,
                 channel_shift_range=0.,
                 fill_mode='constant',
                 cval=0.,
                 label_cval=255,
                 pad_size=None,
                 horizontal_flip=False,
                 vertical_flip=False
                 ):
        data_format = K.image_data_format()
        self.rotation_range=rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.shear_range = shear_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.label_cval=label_cval
        self.zoom_maintain_shape = zoom_maintain_shape

        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

        self.data_format = data_format
        self.channel_index = 3
        self.row_index = 1
        self.col_index = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

    def flow_from_directory(self, file_path, data_dir, data_suffix,
                            label_dir, label_suffix, classes,
                            ignore_label=255,
                            target_size=None, color_mode='rgb',
                            class_mode='sparse',
                            batch_size=32, shuffle=True, seed=None,
                            save_to_dir=None, save_prefix='', save_format='jpeg',
                            loss_shape=None):
        return SegDirectoryIterator(
            file_path, self,
            data_dir=data_dir, data_suffix=data_suffix,
            label_dir=label_dir, label_suffix=label_suffix,
            classes=classes, ignore_label=ignore_label,
            label_cval=self.label_cval,
            target_size=target_size, color_mode=color_mode,
            class_mode=class_mode,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir, save_prefix=save_prefix,
            save_format=save_format,
            loss_shape=loss_shape)


    def random_transform(self, x, y, crop_size):
        # pylint: disable=no-member
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        assert x.shape[img_row_index] == y.shape[img_row_index] and x.shape[img_col_index] == y.shape[
            img_col_index], 'DATA ERROR: Different shape of data and label!\ndata shape: %s, label shape: %s' % (str(x.shape), str(y.shape))

        # use composition of homographies to generate final transform that
        # needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * \
                np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            # * x.shape[img_row_index]
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range) * crop_size[0]
        else:
            tx = 0

        if self.width_shift_range:
            # * x.shape[img_col_index]
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range) * crop_size[1]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 2)
        if self.zoom_maintain_shape:
            zy = zx
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(
            np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)

        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        y = apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode='constant', cval=self.label_cval)


        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)

        x, y = pair_random_crop(x, y, crop_size, self.data_format)

        return x, y

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
import os


def extract_data(filename, num_data, head_size, data_size):
	with gzip.open(filename) as bytestream:
		bytestream.read(head_size)
		buf = bytestream.read(data_size * num_data)
		data = np.frombuffer(buf, dtype=np.uint8).astype(np.float)
	return data


def check_folder(log_dir):
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	return log_dir


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, crop=True, grayscale=False):
	image = imread(image_path, grayscale)
	return transform(image, input_height, input_width, resize_height, resize_width, crop)


def save_images(images, size, image_path):
	return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
	if (grayscale):
		return scipy.misc.imread(path, flatten=True).astype(np.float)
	else:
		return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
	return inverse_transform(images)


def merge(images, size):
	h, w = images.shape[1], images.shape[2]
	if (images.shape[3] in (3, 4)):
		c = images.shape[3]
		img = np.zeros((h * size[0], w * size[1], c))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w, :] = image
		return img
	elif images.shape[3] == 1:
		img = np.zeros((h * size[0], w * size[1]))
		for idx, image in enumerate(images):
			i = idx % size[1]
			j = idx // size[1]
			img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
		return img
	else:
		raise ValueError('in merge(images,size) images parameter ''must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
	image = np.squeeze(merge(images, size))
	return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
	if crop_w is None:
		crop_w = crop_h
	h, w = x.shape[:2]
	j = int(round((h - crop_h) / 2.))
	i = int(round((w - crop_w) / 2.))
	return scipy.misc.imresize(x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width, resize_height=64, resize_width=64, crop=True):
	if crop:
		cropped_image = center_crop(image, input_height, input_width, resize_height, resize_width)
	else:
		cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
	return np.array(cropped_image) / 127.5 - 1.


def inverse_transform(images):
	return (images + 1.) / 2.

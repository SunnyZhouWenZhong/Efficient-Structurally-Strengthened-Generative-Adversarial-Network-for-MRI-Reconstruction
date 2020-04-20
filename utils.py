from tensorlayer.prepro import *
import numpy as np
import skimage.measure
import scipy
from time import localtime, strftime
import logging
import tensorflow as tf
import os


def distort_img(x):
	x = (x + 1.) / 2.
	x = flip_axis(x, axis=1, is_random=True)
	x = elastic_transform(x, alpha=255 * 3, sigma=255 * 0.10, is_random=True)
	x = rotation(x, rg=10, is_random=True, fill_mode='constant')
	x = shift(x, wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
	x = zoom(x, zoom_range=(0.90, 1.10), fill_mode='constant')
	x = brightness(x, gamma=0.05, is_random=True)
	x = x * 2 - 1
	return x


def to_bad_img(x, mask):
	x = (x + 1.) / 2.
	fft = scipy.fftpack.fft2(x[:, :, 0])
	fft = scipy.fftpack.fftshift(fft)
	fft = fft * mask
	fft = scipy.fftpack.ifftshift(fft)
	x = scipy.fftpack.ifft2(fft)
	x = np.abs(x)
	x = x * 2 - 1
	return x[:, :, np.newaxis]


def fft_abs_for_map_fn(x):
	x = (x + 1.) / 2.
	x_complex = tf.complex(x, tf.zeros_like(x))[:, :, 0]
	fft = tf.spectral.fft2d(x_complex)
	fft_abs = tf.abs(fft)
	return fft_abs

def ssim(data):
	x_good, x_bad = data
	x_good = np.squeeze(x_good)
	x_bad = np.squeeze(x_bad)
	ssim_res = skimage.measure.compare_ssim(x_good, x_bad)
	return ssim_res

def psnr(data):
	x_good, x_bad = data
	psnr_res = skimage.measure.compare_psnr(x_good, x_bad)
	return psnr_res

def _tf_fspecial_gauss(size, sigma):
	"""Function to mimic the 'fspecial' gaussian MATLAB function
	"""
	x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

	x_data = np.expand_dims(x_data, axis=-1)
	x_data = np.expand_dims(x_data, axis=-1)

	y_data = np.expand_dims(y_data, axis=-1)
	y_data = np.expand_dims(y_data, axis=-1)

	x = tf.constant(x_data, dtype=tf.float32)
	y = tf.constant(y_data, dtype=tf.float32)

	g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
	return g / tf.reduce_sum(g)

def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
	window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
	K1 = 0.01
	K2 = 0.03
	L = 1  # depth of image (255 in case the image has a differnt scale)
	C1 = (K1*L)**2
	C2 = (K2*L)**2
	mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
	mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
	mu1_sq = mu1*mu1
	mu2_sq = mu2*mu2
	mu1_mu2 = mu1*mu2
	sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
	sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
	sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
	if cs_map:
		value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
					(sigma1_sq + sigma2_sq + C2)),
				(2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
	else:
		value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
					(sigma1_sq + sigma2_sq + C2))

	if mean_metric:
		value = tf.reduce_mean(value)
	return value

def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
	weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
	mssim = []
	mcs = []
	for l in range(level):
		ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
		mssim.append(tf.reduce_mean(ssim_map))
		mcs.append(tf.reduce_mean(cs_map))
		filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
		filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
		img1 = filtered_im1
		img2 = filtered_im2

	# list to tensor of dim D+1
	mssim = tf.stack(mssim, axis=0)
	mcs = tf.stack(mcs, axis=0)

	value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
							(mssim[level-1]**weight[level-1]))

	if mean_metric:
		value = tf.reduce_mean(value)
	return value

def count_trainable_params(scope):
	total_parameters = 0
	for variable in tf.trainable_variables(scope=scope):
		shape = variable.get_shape()
		variable_parametes = 1
		for dim in shape:
			variable_parametes *= dim.value
		total_parameters += variable_parametes
	return total_parameters

def logging_setup(log_dir):
	current_time_str = strftime("%Y_%m_%d_%H_%M_%S", localtime())
	log_all_filename = os.path.join(log_dir, 'log_all_{}.log'.format(current_time_str))
	log_eval_filename = os.path.join(log_dir, 'log_eval_{}.log'.format(current_time_str))

	log_all = logging.getLogger('log_all')
	log_all.setLevel(logging.DEBUG)
	log_all.addHandler(logging.FileHandler(log_all_filename))

	log_eval = logging.getLogger('log_eval')
	log_eval.setLevel(logging.INFO)
	log_eval.addHandler(logging.FileHandler(log_eval_filename))

	log_50_filename = os.path.join(log_dir, 'log_50_images_testing_{}.log'.format(current_time_str))

	log_50 = logging.getLogger('log_50')
	log_50.setLevel(logging.DEBUG)
	log_50.addHandler(logging.FileHandler(log_50_filename))

	return log_all, log_eval, log_50, log_all_filename, log_eval_filename, log_50_filename


if __name__ == "__main__":
	pass

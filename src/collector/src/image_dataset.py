#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: zhangping
# Create at : 2017.12.25
# Description: manage train dataset for neural action planner.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
import gzip
import tensorflow as tf
import pdb
#from feature_extracter import feature_extracter

IMAGES_MAX_NUM = 100000
IMAGES_MAX_ROWS = 1000
IMAGES_MAX_COLS = 1000


def _read32(bytestream):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def _write32(bytestream, int_num):
  dt = numpy.dtype(numpy.uint32).newbyteorder('>')
  buf = numpy.array(int_num,dtype=dt).tobytes()
  return bytestream.write(buf)


def extract_images(filename):
  """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
  print('Extracting', filename)
  with tf.gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2051:
      raise ValueError(
          'Invalid magic number %d in MNIST image file: %s' %
          (magic, filename))
    num_images = _read32(bytestream)
    rows = _read32(bytestream)
    cols = _read32(bytestream)
    buf = bytestream.read(rows * cols * num_images)
    data = numpy.frombuffer(buf, dtype=numpy.uint8)
    data = data.reshape(num_images, rows, cols, 1)
    return data


#def extract_images_and_transform(filename, dims):
#  """Extract the images into a 4D uint8 numpy array [index, y, x, depth], as well as 
#  transform them to grid feature.
#
#  Args:
#    filename: file name of image dataset.
#    dims: feature dimension.
#
#  Return:
#    data of grid feature. A numpy array. shape is [index, y, x, depth].
#  """
#  print('Extracting and transform ', filename)
#  with tf.gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
#    magic = _read32(bytestream)
#    if magic != 2051:
#      raise ValueError(
#          'Invalid magic number %d in MNIST image file: %s' %
#          (magic, filename))
#    # Read header
#    num_images = _read32(bytestream)
#    rows = _read32(bytestream)
#    cols = _read32(bytestream)
#    assert num_images>=0 and num_images<IMAGES_MAX_NUM, (
#          'Error: invalid images number, %d' % (num_images))
#    assert rows>=0 and rows<IMAGES_MAX_ROWS, (
#          'Error: invalid images rows, %d' % (rows))
#    assert cols>=0 and cols<IMAGES_MAX_COLS, (
#          'Error: invalid images colums, %d' % (cols))
#    # Read and transform images
#    data = numpy.zeros([num_images,rows,cols,1])
#    for i in xrange(num_images):
#      image_buf = bytestream.read(rows * cols)
#      image_data = numpy.frombuffer(image_buf, dtype=numpy.uint8)
#      image = image_data.reshape(1, rows, cols, 1)
#      feature = feature_extracter(image,dims)
#      data[i,:,:,0] = feature[0,:,:,0]
#
#    return data


def write_images(filename, images, img_size):
  """Write the images into a file.
  Args:
      filename:
      images: type is list(?numpy.array).
  """
  print('Writing', filename)
  with tf.gfile.Open(filename, 'wb') as f, gzip.GzipFile(fileobj=f,mode='wb') as bytestream:
    magic = 2051
    num_images = images.shape[0]
    rows = img_size[0]
    cols = img_size[1]
    dep = 1

    _write32(bytestream,magic)
    _write32(bytestream,num_images)
    _write32(bytestream,rows)
    _write32(bytestream,cols)

    for i in xrange(0,num_images):
      img = images[i]*255
      img_int = img.astype(numpy.uint8)
      buf = img_int.tobytes()
      bytestream.write(buf)


def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


def extract_labels(filename, one_hot=False, num_classes=10):
  """Extract the labels into a 1D uint8 numpy array [index]."""
  print('Extracting ', filename)
  with tf.gfile.Open(filename, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream:
    magic = _read32(bytestream)
    if magic != 2049:
      raise ValueError(
          'Invalid magic number %d in MNIST label file: %s' %
          (magic, filename))
    num_items = _read32(bytestream)
    buf = bytestream.read(num_items)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    if one_hot:
      return dense_to_one_hot(labels, num_classes)
    return labels


def write_labels(filename, labels):
  """Write the labels into a file.
  Args:
      filename:
      labels: type is list.
  """
  print('Writing', filename)
  with tf.gfile.Open(filename, 'wb') as f, gzip.GzipFile(fileobj=f,mode='wb') as bytestream:
    magic = 2049
    num_items = labels.shape[0]

    _write32(bytestream,magic)
    _write32(bytestream,num_items)
    labels_int = labels.astype(numpy.uint8)
    buf = labels_int.tobytes()
    bytestream.write(buf)


class DataSet(object):

  def __init__(self, images, labels, fake_data=False, one_hot=False,
               dtype=tf.float32):
    """Construct a DataSet.

    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    if fake_data:
      self._num_examples = 10000
      self.one_hot = one_hot
      self._image_rows = 0
      self._image_cols = 0
    else:
      assert images.shape[0] == labels.shape[0], (
          'Error: images.shape: %s labels.shape: %s' % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      self._image_rows = images.shape[1]
      self._image_cols = images.shape[2]

      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      assert images.shape[3] == 1
      #images = images.reshape(images.shape[0],
      #                        images.shape[1] * images.shape[2])
      if dtype == tf.float32:
        # Convert from [0, 255] -> [0.0, 1.0].
        images = images.astype(numpy.float32)
        images = numpy.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def image_rows(self):
    return self._image_rows

  @image_rows.setter
  def image_rows(self,x):
    self._image_rows = x

  @property
  def image_cols(self):
    return self._image_cols

  @image_cols.setter
  def image_cols(self,x):
    self._image_cols = x

  @property
  def images(self):
    return self._images

  @images.setter
  def images(self, x):
    self._images = x

  @property
  def labels(self):
    return self._labels

  @labels.setter
  def labels(self,x):
    self._labels = x

  @property
  def num_examples(self):
    return self._num_examples

  @num_examples.setter
  def num_examples(self,x):
    self._num_examples = x

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1] * 784
      if self.one_hot:
        fake_label = [1] + [0] * 9
      else:
        fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_dataset(dataset_dir, images_file, labels_file, fake_data=False, one_hot=False, dtype=tf.float32):
  """Read a specified data set.

  Args:

  Return: A DataSet instance.
  """
  if fake_data:
    return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

  images = extract_images(dataset_dir + images_file)
  labels = extract_labels(dataset_dir + labels_file, one_hot=one_hot)

  return DataSet(images, labels, dtype=dtype)


def read_dataset_and_transform(dataset_dir, images_file, dims, labels_file, fake_data=False, one_hot=False, dtype=tf.float32):
  """Read a specified data set.

  Args:
    dataset_dir: file directory. 
    images_file: 
    dims       : feature dimension. i.e [28,28].
    labels_file: 
    fake_data  : use fake data.
    one_hot    : one-hot encoding.
  Return: A DataSet instance.
  """
  if fake_data:
    return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

  images = extract_images_and_transform(dataset_dir + images_file,dims)
  labels = extract_labels(dataset_dir + labels_file, one_hot=one_hot)

  return DataSet(images,labels, dtype=dtype)


def save_dataset(save_dir, images_file, labels_file, dataset):
  """Save a specified data set.
  Args:
      save_dir: file path.
      images_file: file name for images.
      labels_file: file name for labels.
      dataset: data set.
  """
  if not tf.gfile.Exists(save_dir):
    print ("Create directory ",save_dir)
    tf.gfile.MakeDirs(save_dir)

  img_size = [dataset.image_rows, dataset.image_cols]
  write_images(save_dir + images_file,dataset.images,img_size)
  write_labels(save_dir + labels_file,dataset.labels)

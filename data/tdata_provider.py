from abc import abstractmethod
import numpy as np
import torch
import h5py
from os.path import join
from torch.utils.data import Dataset
import random

_SEED = 5


class BrainDataProvider(Dataset):

    def __init__(self, h5py_path, config):
        """
        Data provider for multimodal brain MRI.
        Usage:
        :param tfrecord_path: path to tfrecord
        :param config: dict to config options
        """
        f = h5py.File(join(h5py_path, 'det_data.h5py'))
        subject_list = list(self._h5py_file.keys())

        self._h5py_path = h5py_path
        self._h5py_file = f
        self._mode = config.get('mode')
        self._shuffle = config.get('shuffle', True)
        self._subjects = np.asarray(subject_list)
        self._subjects = [i[4:] for i in self._subjects]
        self._nb_folds = config.get('nb-folds', 10)
        

        fold_length = len(self._subjects) // self._nb_folds
        self._subjects = self._rotate(self._subjects, config.get('fold', 0) * fold_length)
        train_idx = (self._nb_folds - 2) * fold_length
        valid_idx = (self._nb_folds - 1) * fold_length
        if self._mode == 'train':
            self._subjects = self._subjects[:train_idx]
        elif self._mode == 'valid':
            self._subjects = self._subjects[train_idx:valid_idx]
        elif self._mode == 'test':
            self._subjects = self._subjects[valid_idx:]
        self._subjects = torch.tensor(np.asarray(self._subjects, np.int32))

    def __len__(self):
        return len(self._subjects)

    def __getitem__(self, idx):
        subj_list = list(self._h5py_file.keys())
        subj = subj_list[idx]
        tp_list = list(self._h5py_file[subj].keys())
        tp = tp_list
        subj = [i[4:] for i in subj]
        return img, subj, tp, label

    @staticmethod
    def _process_data(x):
        x = tf.transpose(x, [3, 2, 1, 0])  # (bs, z, y, x, modality)
        x = x[12:204, 12:204, :, :]
        # x = tf.clip_by_value(x, 0., 1.)
        x = tf.pad(x, paddings=tf.constant([[0, 0], [0, 0], [2, 2], [0, 0]]),
                   mode='CONSTANT', constant_values=0)
        return x

    @staticmethod
    def _rotate(l, n):
        return l[-n:] + l[:-n]

    def _subj_filter(self, subj, tp, img, label):
        return tf.reduce_any(tf.equal(subj, self._subjects))

    def _remove_subj_tp(self, subj, tp, img, label):
        img = tf.decode_raw(img, tf.float32)
        label = tf.cast(tf.decode_raw(label, tf.int16), tf.float32)
        img = tf.reshape(img, tf.stack([4, 60, 256, 256]))
        label = tf.reshape(label, tf.stack([1, 60, 256, 256]))
        # label = tf.expand_dims(tf.reshape(label, tf.stack([60, 256, 256])), 0)
        img = self._process_data(img)  # [..., :-1]
        label = self._process_data(label)
        return img, label

    @abstractmethod
    def get_generator(self, batch_size, nb_epochs):
        raise NotImplementedError


class BrainVolumeDataProvider(BrainDataProvider):

    def get_generator(self, batch_size, nb_epochs):
        """
        Generator that yields examples (subj, time_point, x_batch, y_batch) for use during model training
        :param batch_size: batch size of examples to yield
        :param nb_epochs: number of epochs to generate
        """
        dataset = BrainDataProvider([self.h5py_path], config)
        for i in range
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=10)
        dataset = dataset.filter(self._subj_filter)
        dataset = dataset.map(self._remove_subj_tp, num_parallel_calls=10)
        if self._shuffle:
            dataset = dataset.shuffle(buffer_size=400, seed=_SEED)
        dataset = dataset.repeat(nb_epochs)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(50)
        iterator = dataset.make_initializable_iterator()
        return iterator

    def get_test_generator(self, batch_size=1, nb_epochs=1):
        """
        Generator that yields examples (subj, time_point, x_batch, y_batch) for use during model evaluation
        :param batch_size: batch size of examples to yield
        :param nb_epochs: number of epochs to generate
        """
        dataset = tf.data.TFRecordDataset(self._tfrecord)
        dataset = dataset.map(self._parse_tfrecord).filter(self._subj_filter)
        dataset = dataset.repeat(nb_epochs)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        return iterator

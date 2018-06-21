import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.io
import cv2
import os
import pickle
from collections import defaultdict


def preprocess(x, n_bits_x=None, rand=True):
    x = tf.cast(x, 'float32')
    if n_bits_x < 8:
        x = tf.floor(x / 2 ** (8 - n_bits_x))
    n_bins = 2. ** n_bits_x
    # add [0, 1] random noise
    if rand:
        x = x + tf.random_uniform(tf.shape(x), 0., 1.)
    x = x / n_bins - .5
    return x


def postprocess(x, n_bits_x=None):
    n_bins = 2. ** n_bits_x
    x = tf.floor((x + .5) * n_bins) * (2 ** (8 - n_bits_x))
    return tf.cast(tf.clip_by_value(x, 0, 255), 'uint8')


def split_dataset(xs, ys, n_labels, seed=1234):
    data_dict = defaultdict(list)
    for x, y in zip(xs, ys):
        data_dict[y].append(x)
    np.random.seed(seed)
    xs_u = []
    xs_l = []
    ys_l = []
    ys_u = []
    n_class = len(data_dict.keys())
    assert n_labels % n_class == 0, "num class must divide num labels"
    n_per_class = n_labels // n_class
    for y in data_dict.keys():
        cur_xs = data_dict[y]
        np.random.shuffle(cur_xs)
        cur_xs_l = cur_xs[:n_per_class]
        cur_xs_u = cur_xs[n_per_class:]
        xs_u.extend(cur_xs_u)
        xs_l.extend(cur_xs_l)
        ys_l.extend([y] * n_per_class)
        ys_u.extend([y] * len(cur_xs_u))
    xs_l = np.array(xs_l, dtype=xs.dtype)
    xs_u = np.array(xs_u, dtype=xs.dtype)
    ys_l = np.array(ys_l, dtype=ys.dtype)
    ys_u = np.array(ys_u, dtype=ys.dtype)
    return xs_u, ys_u, xs_l, ys_l


def create_dataset(x, y, batch_size, shuffle=True, repeat=False, ind_aug=None, batch_aug=None):
    ds_x = tf.data.Dataset.from_tensor_slices(x)
    if ind_aug is not None:
        ds_x = ds_x.map(ind_aug)

    if y is None:
        ds = ds_x
    else:
        ds_y = tf.data.Dataset.from_tensor_slices(y)
        ds = tf.data.Dataset.zip((ds_x, ds_y))

    if shuffle:
        ds = ds.shuffle(len(x))

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size)

    if batch_aug is not None:
        ds = ds.map(batch_aug)

    return ds


class Dataset(object):
    def batch_aug(self, x, y):
        return preprocess(x, self.n_bits_x, True), y

    def batch_aug_unsup(self, x):
        return preprocess(x, self.n_bits_x, True)

    def __init__(self, trainx, trainy, testx, testy, batch_size,
                 valx=None, valy=None,
                 train_aug=lambda x: x, test_aug=lambda x: x,
                 init_size=None, n_labels=None, n_valid=None, n_bits_x=None):
        self.n_bits_x = n_bits_x

        # create validation set if requested
        if n_valid is not None:
            # ensure that we are not given a validation set if we are asked to make one
            assert valx is None and valy is None, "If you want me to make a validation set, then don't give me one"
            trainx, trainy, valx, valy = split_dataset(trainx, trainy, n_valid)

        # store original trainx so we can use it go generate a large init batch
        # since no labels are used in initialization, this is ok
        x_orig, y_orig = trainx, trainy
        # if using unlabeled data sample labeled batches of size bs_l and unlabeled batches of size bs_u
        # such that bs_u + bs_l = batch_size
        if n_labels is not None:
            trainx_unlabeled, _, trainx, trainy = split_dataset(trainx, trainy, n_labels)
            train_u = create_dataset(trainx_unlabeled, None, batch_size,
                                     ind_aug=train_aug, batch_aug=self.batch_aug_unsup)
            iterator_u = tf.data.Iterator.from_structure(train_u.output_types, train_u.output_shapes)
            self.x_u = iterator_u.get_next()
            use_train_u = iterator_u.make_initializer(train_u)
            self.n_train_u = len(trainx_unlabeled)
            # if we are using unlabeled data, we use the unlabeled set to tell us when an epoch has ended
            # since the labeled dataset is much smaller, loop through the training set indefinitely and
            # we use the unlabeled set to tell us when an epoch has ended
            bs_l = min(len(trainx), batch_size)
            train_repeat = True
        else:
            train_repeat = False
            bs_l = batch_size
            self.x_u = None

        self.n_train_l = len(trainx)
        train = create_dataset(trainx, trainy, bs_l,
                               repeat=train_repeat, ind_aug=train_aug, batch_aug=self.batch_aug)
        test = create_dataset(testx, testy, batch_size, shuffle=False, ind_aug=test_aug, batch_aug=self.batch_aug)
        iterator = tf.data.Iterator.from_structure(train.output_types, train.output_shapes)
        self.x, self.y = iterator.get_next()
        self.use_train = iterator.make_initializer(train)
        # if using unlabeled data, group reset ops for labeled and unlabeled training data
        if n_labels is not None:
            self.use_train = tf.group([self.use_train, use_train_u])

        self.use_test = iterator.make_initializer(test)
        if valx is not None:
            valid = create_dataset(valx, valy, batch_size,
                                   shuffle=False, ind_aug=test_aug, batch_aug=self.batch_aug)
            self.use_valid = iterator.make_initializer(valid)
        else:
            self.use_valid = None

        if init_size is not None:
            init = create_dataset(x_orig, y_orig, init_size, ind_aug=train_aug, batch_aug=self.batch_aug)
            self.use_init = iterator.make_initializer(init)


class MNISTDataset(Dataset):
    def __init__(self, batch_size, init_size=None, n_labels=None, n_valid=None, n_bits_x=None):
        assert n_valid is None
        self.n_class = 10
        def train_aug(x):
            x = tf.image.resize_image_with_crop_or_pad(x, 36, 36)
            x = tf.random_crop(x, [32, 32, 1])
            return x
        def test_aug(x):
            return tf.image.resize_image_with_crop_or_pad(x, 32, 32)
        cvt = lambda x: ((255 * x).astype(np.uint8)).reshape([-1, 28, 28, 1])
        mnist = input_data.read_data_sets("MNIST_data")
        trainx = cvt(mnist.train.images)
        valx = cvt(mnist.validation.images)
        testx = cvt(mnist.test.images)

        super(MNISTDataset, self).__init__(
            trainx, mnist.train.labels,
            testx,  mnist.test.labels,
            batch_size,
            valx=valx, valy=mnist.validation.labels,
            train_aug=train_aug, test_aug=test_aug,
            init_size=init_size, n_labels=n_labels, n_bits_x=n_bits_x
        )


class CIFAR10Dataset(Dataset):
    def __init__(self, batch_size, init_size=None, n_labels=None, n_valid=None, n_bits_x=None):
        self.n_class = 10
        def load(f):
            with open(f, 'rb') as f:
                stuff = pickle.load(f, encoding="bytes")
                return stuff[b'data'], stuff[b'labels']

        dname = 'cifar-10-batches-py'
        tr_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        tr_names = [os.path.join(dname, tr) for tr in tr_names]
        te_name = os.path.join(dname, "test_batch")
        train_data = [load(f) for f in tr_names]
        trainx = [td[0] for td in train_data]
        trainy = [td[1] for td in train_data]
        trainx = np.concatenate(trainx)
        trainy = np.concatenate(trainy)
        testx, testy = load(te_name)
        trainx = trainx.reshape([-1, 3, 32, 32])
        testx = testx.reshape([-1, 3, 32, 32])
        trainx = np.transpose(trainx, [0, 2, 3, 1])
        testx = np.transpose(testx, [0, 2, 3, 1])
        trainy = np.array(trainy, dtype=np.uint8)
        testy = np.array(testy, dtype=np.uint8)

        def train_aug(x):
            x = tf.image.random_flip_left_right(x)
            x = tf.pad(x, [[4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
            x = tf.random_crop(x, [32, 32, 3])
            return x

        super(CIFAR10Dataset, self).__init__(
            trainx, trainy,
            testx, testy,
            batch_size,
            train_aug=train_aug, init_size=init_size, n_labels=n_labels, n_valid=n_valid, n_bits_x=n_bits_x
        )


class SVHNDataset(Dataset):
    def __init__(self, batch_size, init_size=None, n_labels=None, n_valid=None, n_bits_x=None):
        self.n_class = 10
        train = scipy.io.loadmat("SVHN_data/train_32x32.mat")
        trainx, trainy = train['X'], train['y'][:, 0] - 1
        trainx = trainx.transpose((3, 0, 1, 2))
        test = scipy.io.loadmat("SVHN_data/train_32x32.mat")
        testx, testy = test['X'], test['y'][:, 0] - 1
        testx = testx.transpose((3, 0, 1, 2))

        def train_aug(x):
            x = tf.pad(x, [[4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
            x = tf.random_crop(x, [32, 32, 3])
            return x

        super(SVHNDataset, self).__init__(
            trainx, trainy,
            testx, testy,
            batch_size,
            train_aug=train_aug, init_size=init_size, n_labels=n_labels, n_valid=n_valid, n_bits_x=n_bits_x
        )


def gs(x):
    return x.get_shape().as_list()


def normal_logpdf(x, mu, logvar):
    logp = -.5 * (np.log(2. * np.pi) + logvar + ((x - mu) ** 2) / tf.exp(logvar))
    return logp


def mog_sample(mus, shape, stddev=1.):
    n_class = gs(mus)[0]
    inds = tf.one_hot(tf.argmax(tf.random_uniform([shape[0], n_class]), axis=1), n_class)
    chosen_mus = tf.reduce_sum(mus[None, :, :, :, :] * inds[:, :, None, None, None], axis=1)
    samples = tf.random_normal(shape, stddev=stddev) + chosen_mus
    return samples


if __name__ == "__main__":
    sess = tf.Session()

    dataset = MNISTDataset(
        128, n_labels=100, n_bits_x=5
    )


    # mus = tf.random_normal([2, 5, 5, 3])
    # z = tf.random_normal([13, 5, 5, 3])
    # shape = tf.shape(z)
    # sample = mog_sample(mus, shape)
    # s = sess.run(sample)
    # print(s.shape)
    # mnist = input_data.read_data_sets("MNIST_data")
    # # load data and convert to char
    # cvt = lambda x: ((255 * x).astype(np.uint8)).reshape([-1, 28, 28, 1])
    # data = cvt(mnist.train.images)
    # labels = mnist.train.labels
    # data_test = cvt(mnist.test.images)
    #
    # data = np.array([data[0] for i in range(100)])
    # labels = np.array([labels[0] for i in range(100)])
    #
    # dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    # dataset = dataset.map(lambda x, y: (tf.image.resize_image_with_crop_or_pad(x, 36, 36), y))
    # print(dataset)
    # dataset = dataset.map(lambda x, y: (tf.random_crop(x, [32, 32, 1]), y))
    # print(dataset)
    # dataset = dataset.shuffle(data.shape[0])
    # dataset = dataset.batch(10)
    # iterator = dataset.make_initializable_iterator()
    # item = iterator.get_next()[0]
    # print(iterator, item)
    #
    # 1/0
    #
    # dataset = CIFAR10Dataset(10)

    x, y, xu = dataset.x, dataset.y, dataset.x_u
    sess.run(dataset.use_train)
    _x, _y, _xu = sess.run([x, y, xu])
    print(_x.shape, _y.shape, _xu.shape)
    1.0
    for (im, l) in zip(_x, _y, _xu):
        print(l)
        cv2.imshow('im', im)
        cv2.waitKey(0)
    1/0
    print(_x, _y, _u)
    sess.run(dataset.use_init)
    _x, _y = sess.run([x, y])
    print(_x, _y)
    sess.run(dataset.use_valid)
    _x, _y = sess.run([x, y])
    print(_x, _y)
    sess.run(dataset.use_test)
    _x, _y = sess.run([x, y])
    print(_x, _y)
    #
    # for i in range(100):
    #     _x, _y = sess.run([x, y])
    #     print(_x, _y)
    #     # while True:
    #     #     try:
    #     #         res = sess.run(x)
    #     #         for im in res:
    #     #             cv2.imshow("im", im)
    #     #             cv2.waitKey(0)
    #     #         print('batch', res.shape)
    #     #     except tf.errors.OutOfRangeError:
    #     #         print("reinit")
    #     #         break
    #
    #     # for im in res:
    #     #     cv2.imshow("im", im)
    #     #     cv2.waitKey(0)

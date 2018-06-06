import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.io
import cv2
import os
import pickle


class Dataset(object):
    def __init__(self, trainx, trainy, testx, testy, batch_size,
                 valx=None, valy=None,
                 train_aug=lambda x: x, test_aug=lambda x: x, init_size=None):
        self.train = self._create(trainx, trainy, train_aug, batch_size)
        self.test = self._create(testx, testy, test_aug, batch_size)
        self.iterator = tf.data.Iterator.from_structure(self.train.output_types, self.train.output_shapes)
        self.use_train = self.iterator.make_initializer(self.train)
        self.use_test = self.iterator.make_initializer(self.test)
        if valx is not None:
            self.valid = self._create(valx, valy, test_aug, batch_size)
            self.use_valid = self.iterator.make_initializer(self.valid)
        else:
            self.valid = None

        if init_size is not None:
            self.init = self._create(trainx, trainy, train_aug, init_size)
            self.use_init = self.iterator.make_initializer(self.init)

    def _create(self, x, y, aug, batch_size):
        ds_x = tf.data.Dataset.from_tensor_slices(x)
        ds_x = ds_x.map(aug)
        if y is None:
            ds = ds_x
        else:
            ds_y = tf.data.Dataset.from_tensor_slices(y)
            ds = tf.data.Dataset.zip((ds_x, ds_y))

        ds = ds.shuffle(len(x))
        ds = ds.batch(batch_size)
        return ds


class MNISTDataset(Dataset):
    def __init__(self, batch_size, init_size=None):
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
            train_aug=train_aug, test_aug=test_aug, init_size=init_size
        )
        self.n_class = 10


class CIFAR10Dataset(Dataset):
    def __init__(self, batch_size, init_size=None):
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
        #trainx = trainx[:, :, :, ::-1]
        #testx = testx[:, :, :, ::-1]
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
            train_aug=train_aug, init_size=init_size
        )
        self.n_class = 10


class SVHNDataset(Dataset):
    def __init__(self, batch_size, init_size=None):
        train = scipy.io.loadmat("SVHN_data/train_32x32.mat")
        trainx, trainy = train['X'], train['y']
        trainx = trainx.transpose((3, 0, 1, 2))
        test = scipy.io.loadmat("SVHN_data/train_32x32.mat")
        testx, testy = test['X'], test['y']
        testx = testx.transpose((3, 0, 1, 2))

        def train_aug(x):
            x = tf.pad(x, [[4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
            x = tf.random_crop(x, [32, 32, 3])
            return x

        super(SVHNDataset, self).__init__(
            trainx, trainy,
            testx, testy,
            batch_size,
            train_aug=train_aug, init_size=init_size
        )
        self.n_class = 10


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
    mus = tf.random_normal([2, 5, 5, 3])
    z = tf.random_normal([13, 5, 5, 3])
    shape = tf.shape(z)
    sample = mog_sample(mus, shape)
    s = sess.run(sample)
    print(s.shape)
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

    # dataset = CIFAR10Dataset(10)
    # #dataset = SVHNDataset(10)
    # iterator = dataset.train.make_initializable_iterator()
    # init = iterator.initializer
    # x, y = iterator.get_next()
    # sess = tf.Session()
    # for i in range(100):
    #     print(i)
    #     sess.run(init)
    #     while True:
    #         try:
    #             res = sess.run(x)
    #             for im in res:
    #                 cv2.imshow("im", im)
    #                 cv2.waitKey(0)
    #             print('batch', res.shape)
    #         except tf.errors.OutOfRangeError:
    #             print("reinit")
    #             break
    #
    #     # for im in res:
    #     #     cv2.imshow("im", im)
    #     #     cv2.waitKey(0)

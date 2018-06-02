import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import scipy.io
import cv2


class Dataset(object):
    def __init__(self, trainx, trainy, testx, testy, batch_size,
                 valx=None, valy=None,
                 train_aug=lambda x: x, test_aug=lambda x: x):
        self.train = self._create(trainx, trainy, train_aug, batch_size)
        self.test = self._create(testx, testy, test_aug, batch_size)
        if valx is not None:
            self.valid = self._create(valx, valy, test_aug, batch_size)
        else:
            self.valid = None

    def _create(self, x, y, aug, batch_size):
        inds = list(range(len(x)))
        np.random.shuffle(inds)
        x = x[inds]
        y = y[inds]
        ds_x, ds_y = tf.data.Dataset.from_tensor_slices(x), tf.data.Dataset.from_tensor_slices(y)
        ds_x = ds_x.map(aug)
        ds = tf.data.Dataset.zip((ds_x, ds_y))
        #ds = ds.shuffle(x.shape[0])
        ds = ds.batch(batch_size)
        return ds


class MNISTDataset(Dataset):
    def __init__(self, batch_size):
        def train_aug(x):
            x = tf.image.resize_image_with_crop_or_pad(x, 36, 36)
            x = tf.random_crop(x, [32, 32, 1])
            return x
        def test_aug(x):
            return tf.image.resize_image_with_crop_or_pad(x, 32, 32)
        cvt = lambda x: ((255 * x).astype(np.uint8)).reshape([-1, 28, 28, 1])
        mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
        trainx = cvt(mnist.train.images)
        valx = cvt(mnist.validation.images)
        testx = cvt(mnist.test.images)

        super(MNISTDataset, self).__init__(
            trainx, mnist.train.labels,
            testx,  mnist.test.labels,
            batch_size,
            valx=valx, valy=mnist.validation.labels,
            train_aug=train_aug, test_aug=test_aug
        )


class CIFAR10Dataset(Dataset):
    def __init__(self, batch_size):
        (trainx, trainy), (testx, testy) = tf.keras.datasets.cifar10.load_data()
        trainx = trainx[:, :, :, ::-1]
        testx = testx[:, :, :, ::-1]
        testy = testy.astype(np.uint8)

        def train_aug(x):
            x = tf.image.random_flip_left_right(x)
            x = tf.pad(x, [[4, 4], [4, 4], [0, 0]], 'SYMMETRIC')
            x = tf.random_crop(x, [32, 32, 3])
            return x

        super(CIFAR10Dataset, self).__init__(
            trainx, trainy,
            testx, testy,
            batch_size,
            train_aug=train_aug
        )

class SVHNDataset(Dataset):
    def __init__(self, batch_size):
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
            train_aug=train_aug
        )






if __name__ == "__main__":
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

    #dataset = CIFAR10Dataset(10)
    dataset = SVHNDataset(10)
    init = dataset.train_iterator.initializer
    x, y = dataset.train_iterator.get_next()
    sess = tf.Session()
    for i in range(100):
        print(i)
        sess.run(init)
        while True:
            try:
                res = sess.run(x)
                for im in res:
                    cv2.imshow("im", im)
                    cv2.waitKey(0)
                print('batch', res.shape)
            except tf.errors.OutOfRangeError:
                print("reinit")
                break

        # for im in res:
        #     cv2.imshow("im", im)
        #     cv2.waitKey(0)

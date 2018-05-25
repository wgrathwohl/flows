import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def gs(x):
    return x.get_shape().as_list()


def name_var(x, name):
    return tf.identity(x, name=name)


def squeeze(x, factor=2):
    assert factor >= 1
    if factor == 1: return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    assert height % factor == 0 and width % factor == 0
    x = tf.reshape(x, [-1, height//factor, factor, width//factor, factor, n_channels])
    x = tf.transpose(x, [0, 1, 3, 5, 2, 4])
    x = tf.reshape(x, [-1, height//factor, width//factor, n_channels*factor*factor])
    return x


def flatten(x):
    return tf.reshape(x, [-1, np.prod(gs(x)[1:])])


def mask(x, variety="channel", ind=0):
    assert variety in ("channel", "checker")
    assert ind in (0, 1)
    if variety == "channel":
        nc = gs(x)[-1]
        m = np.zeros([1, 1, 1, nc])
        hnc = nc // 2
        if ind == 0:
            m[:, :, :, :hnc] = 1.
        else:
            m[:, :, :, hnc:] = 1.
    else:
        h, w = gs(x)[1:3]
        m = np.zeros([1, h, w, 1])
        m[:, ::2, :, :] = 1. - m[:, ::2, :, :]
        m[:, :, 1::2, :] = 1. - m[:, :, 1::2, :]
        if ind == 1:
            m = 1. - m
    return m


def NN(x, name, dim=32, init=False):
    def conv(h, d, k, name, init, nonlin=True):
        if nonlin:
            h = tf.layers.conv2d(h, d, (k, k), (1, 1), "same",
                                 name=name, use_bias=False,
                                 kernel_initializer=tf.variance_scaling_initializer())
            h = actnorm(h, name + "_an", init)
            h = tf.nn.relu(h)
            return h
        else:
            h = tf.layers.conv2d(h, d, (k, k), (1, 1), "same",
                                 name=name, use_bias=True,
                                 kernel_initializer=tf.zeros_initializer())
            logscale = tf.get_variable("logscale", [], dtype=tf.float32, initializer=tf.zeros_initializer())
            scale = tf.exp(logscale)
            return scale * h
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        nc = gs(x)[-1]
        h = conv(x, dim, 3, "h1", init)
        h = conv(h, dim, 1, "h2", init)
        h = conv(h, nc,  3, "h3", init, nonlin=False)
        tf.summary.histogram(scope.name, h)
    return h


def coupling_layer(x, b, name, init=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        x, logdet_an = actnorm(x, name + "_an_in", init, logdet=True)
        bx = b * x
        s = NN(bx, "s", init=init)
        t = NN(bx, "t", init=init)
        y = bx + (1. - b) * (x * tf.exp(s) + t)
        logdet_y = tf.reduce_sum(s * (1. - b), axis=[1, 2, 3])
        if not init:
            tf.add_to_collection("logdets", name_var(logdet_y, "clayerlogdet"))
    return y, logdet_an + logdet_y


def coupling_block(x, name, t="channel", init=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        m1 = mask(x, t, 0)
        m2 = mask(x, t, 1)
        y, ld1 = coupling_layer(x, m1, "c1", init=init)
        y, ld2 = coupling_layer(y, m2, "c2", init=init)
        y, ld3 = coupling_layer(y, m1, "c3", init=init)
    return y, ld1 + ld2 + ld3


def scale_block(x, name, init=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        y1, ld1 = coupling_block(x, "channel", "channel", init=init)
        y1 = squeeze(y1)
        y2, ld2 = coupling_block(y1, "checker", "checker", init=init)
    return y2, ld1 + ld2


def actnorm(x, name, init=False, eps=1e-6, logdet=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        t = tf.get_variable("t", (1, 1, 1, gs(x)[-1]), trainable=True)
        logs = tf.get_variable("logs", (1, 1, 1, gs(x)[-1]), trainable=True)
        if init:
            t_init, x_var = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
            logs_init = tf.log(1. / (tf.sqrt(x_var) + eps))
            logsop = logs.assign(logs_init)
            top = t.assign(t_init)
            with tf.control_dependencies([logsop, top]):
                an = (x - t_init) * tf.exp(logs_init)
        else:
            logs = tf.check_numerics(logs, "bad logs")
            t = tf.check_numerics(t, "bad t")
            an = (x - t) * tf.exp(logs)

        if logdet:
            h, w = x.get_shape().as_list()[1:3]
            ld = tf.reduce_sum(logs) * h * w
            if not init:
                tf.add_to_collection("logdets", name_var(ld, "actnormlogdet"))
            return an, ld
        else:
            return an


def net(x, name, depth, init=False):
    h = x
    zs = []
    logdet = 0.
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        for i in range(depth):
            h, ld = scale_block(h, str(i), init=init)
            logdet += ld
            bs, a, b, d = gs(h)
            new_d = d // 2
            h, z = h[:, :, :, :new_d], h[:, :, :, new_d:]
            print(gs(h), gs(z))
            tf.summary.histogram("z{}".format(i), z)
            zs.append(tf.reshape(z, [-1, a * b * new_d]))
        zs.append(tf.reshape(h, [-1, a * b * new_d]))
    z = tf.concat(zs, 1)
    return z, logdet


def normal_logpdf(x, mu, logvar):
    logp = -.5 * (np.log(2. * np.pi) + logvar + ((x - mu) ** 2) / tf.exp(logvar))
    return logp


if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    data = mnist.train.images.reshape([-1, 28, 28, 1])
    data_pad = np.zeros((data.shape[0], 32, 32, 1))
    data_pad[:, 2:-2, 2:-2, :] = data

    data_test = mnist.test.images.reshape([-1, 28, 28, 1])
    data_test_pad = np.zeros((data_test.shape[0], 32, 32, 1))
    data_test_pad[:, 2:-2, 2:-2, :] = data_test

    inds = list(range(data.shape[0]))
    test_inds = list(range(data_test.shape[0]))

    np.random.shuffle(inds)
    init_inds = inds[:1024]
    init_batch = data_pad[init_inds]
    
    sess = tf.Session()


    x = tf.placeholder(tf.float32, [None, 32, 32, 1], name="x")
    xs = tf.tile(x, [1, 1, 1, 3])
    xs = squeeze(xs)
    #
    # m = mask(xs, "channel", 0)
    # z_init, _ = coupling_layer(xs, m, "test_cl", init=True)
    # z, logdet = coupling_layer(xs, m, "test_cl")
    # z = tf.reshape(z, [-1, 16 * 16 * 12])

    # z_init, logdet = coupling_block(xs, "b1", t="channel", init=True)
    # z, _ = coupling_block(xs, "b1", t="channel")
    # z = tf.reshape(z, [-1, 16 * 16 * 12])
    #
    # z_init, _ = scale_block(xs, "s1", True)
    # z, logdet = scale_block(xs, "s1")
    # z = flatten(z)

    z_init, _ = net(xs, "net", 2, init=True)
    z, logdet = net(xs, "net", 2)

    logpz = tf.reduce_mean(tf.reduce_sum(normal_logpdf(z, 0., 0.), axis=1))
    logdet = tf.reduce_mean(logdet)
    total_logprob = logpz + logdet
    loss = -total_logprob / np.prod(gs(xs)[1:])
    lr = tf.placeholder(tf.float32, [], name="lr")
    optim = tf.train.AdamOptimizer(lr)
    opt = optim.minimize(loss)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("logdet", logdet)
    tf.summary.scalar("logp", logpz)

    for v in tf.get_collection("logdets"):
        print(v.name)
        tf.summary.scalar(v.name, tf.reduce_mean(v))
    for v in tf.trainable_variables():
        tf.summary.histogram(v.name, v)
    sum_op = tf.summary.merge_all()

    #LOGDET IS FUXED YO

    # nn = NN(x, "test")
    # print(gs(x), gs(nn))
    #
    # m = mask(x, "checker", 1)
    # print(m[0, :, :, 0])
    #
    # cl, ld = coupling_layer(x, mask(x, "checker", 0), "test_cl")
    # print(gs(cl))
    #
    # s = squeeze(x)
    # print(gs(x), gs(s))

    #
    # z_init, _ = actnorm(x, "an", True, logdet=True)
    # z, logdet = actnorm(x, "an", False, logdet=True)

    sess.run(tf.global_variables_initializer())
    sess.run(z_init, feed_dict={x: init_batch})
    _an, _ld = sess.run([z, logdet], feed_dict={x: init_batch})
    print(_an.min(), _an.max())
    print(_an.mean(), _an.var())
    print(_ld)

    writer = tf.summary.FileWriter("/tmp/train")

    for i in range(10000):
        batch_inds = np.random.choice(inds, 64, replace=False)
        batch = data_pad[batch_inds]
        iter_lr = .0003 # if i > 1000 else (i / 1000.) * .0003
        logp, lgdt, _, sstr = sess.run([logpz, logdet, opt, sum_op], feed_dict={x: batch, lr: iter_lr})
        writer.add_summary(sstr, i)

        # if i % 10 == 0:
        #     batch_inds = np.random.choice(test_inds, 64, replace=False)
        #     test_batch = data_test_pad[batch_inds]
        #     logpt = sess.run(logpz, feed_dict={x: test_batch})
        #     print(i, logp, logpt)
        # else:
        print(i, logp, lgdt)

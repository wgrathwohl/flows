import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def gs(x):
    return x.get_shape().as_list()


def default_initializer(std=0.05):
    return tf.random_normal_initializer(0., std)


def squeeze(x, factor=2, backward=False):
    assert factor >= 1
    if factor == 1: return x
    shape = x.get_shape()
    height = int(shape[1])
    width = int(shape[2])
    n_channels = int(shape[3])
    if backward:
        assert n_channels >= 4 and n_channels % 4 == 0
        x = tf.reshape(x, (-1, height, width, int(n_channels / factor ** 2), factor, factor))
        x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
        x = tf.reshape(x, (-1, int(height * factor), int(width * factor), int(n_channels / factor ** 2)))
    else:
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
    nc = gs(x)[-1]
    m = np.zeros([1, 1, 1, nc])
    if variety == "channel":
        hnc = nc // 2
        if ind == 0:
            m[:, :, :, :hnc] = 1.
        else:
            m[:, :, :, hnc:] = 1.
    else:
        for i in range(nc):
            if i % 4 == 1 or i % 4 == 2:
                m[:, :, :, i] = 1.
        if ind == 1:
            m = 1. - m
    return m


def NN(x, name, dim=32, init=False):
    def conv(h, d, k, name, init, nonlin=True):
        if nonlin:
            h = tf.layers.conv2d(h, d, (k, k), (1, 1), "same",
                                 name=name, use_bias=False,
                                 kernel_initializer=default_initializer())
            h = actnorm(h, name + "_an", init)
            h = tf.nn.relu(h)
            return h
        else:
            h = tf.layers.conv2d(h, d, (k, k), (1, 1), "same",
                                 name=name, use_bias=True,
                                 kernel_initializer=tf.zeros_initializer(),
                                 bias_initializer=tf.zeros_initializer())
            logs = tf.get_variable("logs", shape=[1, 1, 1, d], dtype=tf.float32, initializer=tf.zeros_initializer())
            s = tf.exp(logs)
            return h * s
    with tf.variable_scope(name, reuse=(not init)):
        nc = gs(x)[-1]
        h = conv(x, dim, 3, "h1", init)
        h = conv(h, dim, 1, "h2", init)
        h = conv(h, nc,  3, "h3", init, nonlin=False)
    return h


def coupling_layer(x, b, name, init=False, backward=False, eps=1e-6):
    def get_vars(b, x):
        bx = b * x
        logit_s = NN(bx, "logit_s", init=init) + 2.
        s = tf.sigmoid(logit_s) + eps
        t = NN(bx, "t", init=init)
        return logit_s, s, t

    with tf.variable_scope(name, reuse=(not init)):
        if backward:
            logit_s, s, t = get_vars(b, x)
            x = b * x + (1. - b) * ((x / s) - t)
            x, logdet_an = actnorm(x, name + "_an_in", init, logdet=True, backward=True)
            logdet = tf.reduce_sum(tf.log_sigmoid(logit_s) * (1. - b), axis=[1, 2, 3])
            return x, -logdet + logdet_an
        else:
            x, logdet_an = actnorm(x, name + "_an_in", init, logdet=True, backward=False)
            logit_s, s, t = get_vars(b, x)
            if not init:
                tf.summary.histogram("s/"+name, s)
                tf.summary.histogram("t/"+name, t)
            x = b * x + (1. - b) * ((x + t) * s)
            logdet = tf.reduce_sum(tf.log_sigmoid(logit_s) * (1. - b), axis=[1, 2, 3])
            return x, logdet + logdet_an


def coupling_block(x, name, t="channel", init=False, backward=False):
    with tf.variable_scope(name, reuse=(not init)):
        m1 = mask(x, t, 0)
        m2 = mask(x, t, 1)
        n1, n2, n3 = ("c3", "c2", "c1") if backward else ("c1", "c2", "c3")
        y, ld1 = coupling_layer(x, m1, n1, init=init, backward=backward)
        y, ld2 = coupling_layer(y, m2, n2, init=init, backward=backward)
        y, ld3 = coupling_layer(y, m1, n3, init=init, backward=backward)
    return y, ld1 + ld2 + ld3


def scale_block(x, name, init=False, backward=False):
    with tf.variable_scope(name, reuse=(not init)):
        n1, n2 = ("checker", "channel") if backward else ("channel", "checker")
        y1, ld1 = coupling_block(x, n1, n1, init=init, backward=backward)
        y1 = squeeze(y1, backward=backward)
        y2, ld2 = coupling_block(y1, n2, n2, init=init, backward=backward)
    return y2, ld1 + ld2


def actnorm(x, name, init=False, eps=1e-6, logdet=False, backward=False):
    def compute(x, logs, t):
        if backward:
            return (x * tf.exp(-logs)) - t
        else:
            return (x + t) * tf.exp(logs)
    def ld(x, logs):
        h, w = x.get_shape().as_list()[1:3]
        val = tf.reduce_sum(logs) * h * w
        if backward:
            return -val
        else:
            return val

    with tf.variable_scope(name, reuse=(not init)):
        t = tf.get_variable("t", (1, 1, 1, gs(x)[-1]), trainable=True)
        logs = tf.get_variable("logs", (1, 1, 1, gs(x)[-1]), trainable=True)
        if init:
            x_mean, x_var = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
            logs_init = tf.log(1. / (tf.sqrt(x_var) + eps))
            t_init = -x_mean
            logsop = logs.assign(logs_init)
            top = t.assign(t_init)
            with tf.control_dependencies([logsop, top]):
                an = compute(x, logs_init, t_init)
        else:
            an = compute(x, logs, t)

        if logdet:
            return an, ld(x, logs)
        else:
            return an


def net(x, name, depth, init=False):
    h = x
    zs = []
    logdet = 0.
    with tf.variable_scope(name, reuse=(not init)):
        for i in range(depth):
            h, ld = scale_block(h, str(i), init=init)
            logdet += ld
            if i == depth - 1:
                zs.append(h)
            else:
                bs, a, b, d = gs(h)
                new_d = d // 2
                h, z = h[:, :, :, :new_d], h[:, :, :, new_d:]
                print(gs(h), gs(z))
                zs.append(z)
    return zs, logdet


def net_backwards(zs, name, init=False):
    logdet = 0.
    with tf.variable_scope(name, reuse=(not init)):
        top_z = zs[-1]
        for i  in reversed(range(len(zs))):
            print(gs(top_z))
            h, ld = scale_block(top_z, str(i), init=init, backward=True)
            logdet += ld
            if i == 0:
                return h, logdet
            else:
                next_z = zs[i - 1]
                top_z = tf.concat([h, next_z], axis=3)


def normal_logpdf(x, mu, logvar):
    logp = -.5 * (np.log(2. * np.pi) + logvar + ((x - mu) ** 2) / tf.exp(logvar))
    return logp


def pad_batch(batch):
    bs = batch.shape[0]
    pads = np.random.randint(4, size=(bs, 2))
    padded = []
    for (px, py), im in zip(pads, batch):
        im_pad = np.zeros((32, 32, 1), dtype=np.uint8)
        im_pad[px: px + 28, py: py + 28, :] = im
        padded.append(im_pad)
    return np.array(padded)


def preprocess(x, n_bits_x=5):
    x = tf.cast(x, 'float32')
    if n_bits_x < 8:
        x = tf.floor(x / 2 ** (8 - n_bits_x))
    n_bins = 2. ** n_bits_x
    # add [0, 1] random noise
    x = x + tf.random_uniform(tf.shape(x), 0., 1.)
    x = x / n_bins - .5
    return x


def postprocess(x, n_bits_x=5):
    n_bins = 2. ** n_bits_x
    x = tf.floor((x + .5) * n_bins) * (2 ** (8 - n_bits_x))
    return tf.cast(tf.clip_by_value(x, 0, 255), 'uint8')


if __name__ == "__main__":
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    # load data and convert to char
    cvt = lambda x: ((255 * x).astype(np.uint8)).reshape([-1, 28, 28, 1])
    data = cvt(mnist.train.images)
    data_test = cvt(mnist.test.images)

    inds = list(range(data.shape[0]))
    test_inds = list(range(data_test.shape[0]))

    np.random.shuffle(inds)
    init_inds = inds[:1024]
    init_batch = data[init_inds]
    init_batch = pad_batch(init_batch)
    
    sess = tf.Session()

    x = tf.placeholder(tf.uint8, [None, 32, 32, 1], name="x")
    xpp = preprocess(x)
    xs = squeeze(xpp)

    z_init, _ = net(xs, "net", 3, init=True)
    z, logdet_orig = net(xs, "net", 3)
    z_samp = [tf.random_normal(tf.shape(_z)) for _z in z]
    xs_recons, logdet_recons = net_backwards(z, "net")
    xs_samp, logdet_samp = net_backwards(z_samp, "net")

    for i, _z in enumerate(z):
        tf.summary.histogram("z{}".format(i), _z)

    x_recons = postprocess(squeeze(xs_recons, backward=True))
    tf.summary.image("x_recons", x_recons)
    tf.summary.image("x", x)
    x_samp = postprocess(squeeze(xs_samp, backward=True))
    tf.summary.image("x_sample", x_samp)
    recons_error = tf.reduce_mean(tf.square(tf.to_float(postprocess(xpp) - x_recons)))

    logpz = tf.add_n([tf.reduce_sum(normal_logpdf(_z, 0., 0.), axis=[1, 2, 3]) for _z in z])
    logpz = tf.reduce_mean(logpz)
    logdet = tf.reduce_mean(logdet_orig)
    total_logprob = (logpz + logdet) - np.log(32.) * np.prod(gs(xs)[1:])
    loss = -total_logprob
    lr = tf.placeholder(tf.float32, [], name="lr")
    optim = tf.train.AdamOptimizer(lr)
    opt = optim.minimize(loss)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("logdet", logdet)
    tf.summary.scalar("logp", logpz)
    tf.summary.scalar("recons", recons_error)
    tf.summary.scalar("lr", lr)


    # for v in tf.trainable_variables():
    #     tf.summary.histogram(v.name, v)
    sum_op = tf.summary.merge_all()


    sess.run(tf.global_variables_initializer())
    sess.run(z_init, feed_dict={x: init_batch})
    train_writer = tf.summary.FileWriter("/tmp/train/train")
    test_writer = tf.summary.FileWriter("/tmp/train/test")

    n_epochs = 10000
    batch_size = 128
    n_data = data.shape[0]
    n_iters = n_data // batch_size
    warm_up_epochs = 10
    warm_up_iters = n_iters * warm_up_epochs
    base_lr = .001


    iter = 0
    for epoch in range(n_epochs):
        np.random.shuffle(inds)
        for i in range(n_iters):
            batch_inds = inds[i * batch_size: (i + 1) * batch_size]
            batch = data[batch_inds]
            batch = pad_batch(batch)
            iter_lr = base_lr * (float(iter) / warm_up_iters) if iter < warm_up_iters else base_lr
            if iter % 100 == 0:
                re, _l, logp, _, sstr = sess.run([recons_error, loss, logpz, opt, sum_op],
                                                 feed_dict={x: batch, lr: iter_lr})
                train_writer.add_summary(sstr, iter)
                test_batch_inds = np.random.choice(test_inds, 128, replace=False)
                test_batch = data_test[test_batch_inds]
                test_batch = pad_batch(test_batch)
                sstr = sess.run(sum_op, feed_dict={x: test_batch, lr: iter_lr})
                test_writer.add_summary(sstr, iter)
                print(i, _l, logp, re)

            else:
                _ = sess.run(sum_op, feed_dict={x: batch, lr: iter_lr})

            iter += 1

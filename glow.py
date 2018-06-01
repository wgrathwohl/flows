import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


def gs(x):
    return x.get_shape().as_list()


def name_var(x, name):
    return tf.identity(x, name=name)


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


def actnorm(x, name, init=False, eps=1e-6, return_logdet=False, backward=False):
    def compute(x, logs, t):
        if backward:
            return (x * tf.exp(-logs)) - t
        else:
            return (x + t) * tf.exp(logs)
    def logdet(x, logs):
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

        if return_logdet:
            return an, logdet(x, logs)
        else:
            return an


def NN(x, name, dim=128, init=False):
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


def split(x):
    nc = gs(x)[-1] // 2
    return x[:, :, :, :nc], x[:, :, :, nc:]


def combine(x1, x2):
    return tf.concat([x1, x2], axis=3)


def coupling_layer(in_feats, name, init=False, backward=False, eps=1e-6):
    def get_vars(feats):
        logit_s = NN(feats, "logit_s", init=init) + 2.
        s = tf.sigmoid(logit_s) + eps
        t = NN(feats, "t", init=init)
        logdet = tf.reduce_sum(tf.log_sigmoid(logit_s), axis=[1, 2, 3])
        return s, t, logdet

    with tf.variable_scope(name):
        if backward:
            y = in_feats
            y1, y2 = split(y)
            s, t, logdet = get_vars(y1)
            x1 = y1
            x2 = (y2 / s) - t
            x = combine(x1, x2)
            return x, -logdet
        else:
            x = in_feats
            x1, x2 = split(x)
            s, t, logdet = get_vars(x1)
            y1 = x1
            y2 = (x2 + t) * s
            y = combine(y1, y2)
            return y, logdet


def random_rotation_matrix(nc):
    return np.linalg.qr(np.random.randn(nc, nc))[0].astype(np.float32)


def invconv_layer(in_feats, name, init=False, backward=False, eps=1e-6):
    conv = lambda f, k: tf.nn.conv2d(f, k, [1, 1, 1, 1], "SAME")
    with tf.variable_scope(name, reuse=(not init)):
        hh, ww, nc = gs(in_feats)[1:]
        w = tf.get_variable("w", dtype=tf.float32, initializer=random_rotation_matrix(nc))
        det = tf.matrix_determinant(w)
        logdet = tf.log(tf.abs(det) + eps) * hh * ww
        if backward:
            y = in_feats
            kernel = tf.reshape(tf.matrix_inverse(w), [1, 1, nc, nc])
            x = conv(y, kernel)
            return x, -logdet
        else:
            x = in_feats
            kernel = tf.reshape(w, [1, 1, nc, nc])
            y = conv(x, kernel)
            return y, logdet


def flow_step(in_feats, name, init=False, backward=False):
    with tf.variable_scope(name, reuse=(not init)):
        if backward:
            y = in_feats
            x, logdet_coupling = coupling_layer(y, "coupling", init=init, backward=True)
            x, logdet_invconv = invconv_layer(x, "invconv", init=init, backward=True)
            x, logdet_actnorm = actnorm(x, "actnorm", init=init, logdet=True, backward=True)
            return x, logdet_actnorm + logdet_invconv + logdet_coupling
        else:
            x = in_feats
            x, logdet_actnorm = actnorm(x, "actnorm", init=init, logdet=True, backward=False)
            x, logdet_invconv = invconv_layer(x, "invconv", init=init, backward=False)
            y, logdet_coupling = coupling_layer(x, "coupling", init=init, backward=False)
            return y, logdet_actnorm + logdet_invconv + logdet_coupling


def scale_block(in_feats, name, flow_steps, init=False, backward=False):
    with tf.variable_scope(name, reuse=(not init)):
        if backward:
            z = in_feats
            logdet = 0.
            for i in reversed(range(flow_steps)):
                z, step_logdet = flow_step(z, "step{}".format(i), init=init, backward=True)
                logdet += step_logdet
            x = squeeze(z, backward=True)
            return x, logdet
        else:
            x = in_feats
            z = squeeze(x, backward=False)
            logdet = 0.
            for i in range(flow_steps):
                z, step_logdet = flow_step(z, "step{}".format(i), init=init, backward=False)
                logdet += step_logdet
            return z, logdet


def net(in_feats, name, num_blocks, flow_steps, init=False, backward=False):
    with tf.variable_scope(name, reuse=(not init)):
        if backward:
            zs = in_feats
            logdet = 0.
            z = zs[-1]
            for i in reversed(range(num_blocks)):
                z, block_logdet = scale_block(z, "block{}".format(i), flow_steps, init=init, backward=True)
                logdet += block_logdet
                if i > 0:
                    zi = zs[i - 1]
                    z = combine(z, zi)
            return z, logdet
        else:
            z = in_feats
            logdet = 0.
            zs = []
            for i in range(num_blocks):
                z, block_logdet = scale_block(z, "block{}".format(i), flow_steps, init=init, backward=False)
                logdet += block_logdet
                if i == num_blocks - 1:
                    zs.append(z)
                else:
                    z, zi = split(z)
                    zs.append(zi)
            return zs, logdet


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
    n_bits_x = 5

    np.random.shuffle(inds)
    init_inds = inds[:1024]
    init_batch = data[init_inds]
    init_batch = pad_batch(init_batch)
    
    sess = tf.Session()

    x = tf.placeholder(tf.uint8, [None, 32, 32, 1], name="x")
    # clip to 5 bits
    xpp = preprocess(x, n_bits_x=n_bits_x)

    z_init, _ = net(xpp, "net", 3, 8, init=True)
    z, logdet_orig = net(xpp, "net", 3, 8)
    z_samp = [tf.random_normal(tf.shape(_z)) for _z in z]
    x_recons, logdet_recons = net(z, "net", 3, 8, backward=True)
    x_samp, logdet_samp = net(z_samp, "net", 3, 8, backward=True)

    for i, _z in enumerate(z):
        tf.summary.histogram("z{}".format(i), _z)

    # visualization for debugging
    x_recons = postprocess(x_recons, n_bits_x=n_bits_x)
    x_samp = postprocess(x_samp, n_bits_x=n_bits_x)
    recons_error = tf.reduce_mean(tf.square(tf.to_float(postprocess(xpp, n_bits_x=n_bits_x) - x_recons)))
    tf.summary.image("x_sample", x_samp)
    tf.summary.image("x_recons", x_recons)
    tf.summary.image("x", x)

    # compute loss
    logpz = tf.add_n([tf.reduce_sum(normal_logpdf(_z, 0., 0.), axis=[1, 2, 3]) for _z in z])
    logpz = tf.reduce_mean(logpz)
    logdet = tf.reduce_mean(logdet_orig)
    total_logprob = (logpz + logdet) - np.log(2.**n_bits_x) * np.prod(gs(xpp)[1:])
    loss = -total_logprob
    bits_x = loss / (np.log(2.) * np.prod(gs(xpp)[1:]))
    lr = tf.placeholder(tf.float32, [], name="lr")
    optim = tf.train.AdamOptimizer(lr)
    opt = optim.minimize(loss)

    # summary for loss, etc
    tf.summary.scalar("loss", loss)
    tf.summary.scalar("logdet", logdet)
    tf.summary.scalar("logp", logpz)
    tf.summary.scalar("recons", recons_error)
    tf.summary.scalar("lr", lr)
    tf.summary.scalar("bits_x", bits_x)
    sum_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())
    # initialize act-norm parameters
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
                print(iter, _l, logp, re)

            else:
                _ = sess.run(sum_op, feed_dict={x: batch, lr: iter_lr})

            iter += 1

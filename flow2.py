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
        if ind == 0:
            m[:, :, :, ::2] = 1.
        else:
            m[:, :, :, 1::2] = 1.
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
                                 kernel_initializer=tf.zeros_initializer())
            return h
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        nc = gs(x)[-1]
        h = conv(x, dim, 3, "h1", init)
        h = conv(h, dim, 1, "h2", init)
        h = conv(h, nc,  3, "h3", init, nonlin=False)
    return h


def coupling_layer(x, b, name, init=False, backward=False):
    def compute(x, b, logs, t):
        logdet = tf.reduce_sum(logs * (1. - b), axis=[1, 2, 3])
        if backward:
            x = b * x + (1. - b) * ((x - t) * tf.exp(-logs))
            x, logdet_an = actnorm(x, name + "_an_in", init, logdet=True, backward=True)
            return x, -logdet + logdet_an
        else:
            x, logdet_an = actnorm(x, name + "_an_in", init, logdet=True, backward=False)
            x = b * x + (1. - b) * (x * tf.exp(logs) + t)
            return x, logdet + logdet_an

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        logs = NN(b * x, "logs", init=init)
        t = NN(b * x, "t", init=init)
        y, logdet = compute(x, b, logs, t)
    return y, logdet


def coupling_block(x, name, t="channel", init=False, backward=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        m1 = mask(x, t, 0)
        m2 = mask(x, t, 1)
        n1, n2, n3 = ("c3", "c2", "c1") if backward else ("c1", "c2", "c3")
        y, ld1 = coupling_layer(x, m1, n1, init=init, backward=backward)
        y, ld2 = coupling_layer(y, m2, n2, init=init, backward=backward)
        y, ld3 = coupling_layer(y, m1, n3, init=init, backward=backward)
    return y, ld1 + ld2 + ld3


def scale_block(x, name, init=False, backward=False):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        n1, n2 = ("checker", "channel") if backward else ("channel", "checker")
        y1, ld1 = coupling_block(x, n1, n1, init=init, backward=backward)
        y1 = squeeze(y1, backward=backward)
        y2, ld2 = coupling_block(y1, n2, n2, init=init, backward=backward)
    return y2, ld1 + ld2


def actnorm(x, name, init=False, eps=1e-6, logdet=False, backward=False):
    def compute(x, logs, t):
        if backward:
            return (x - t) * tf.exp(-logs)
        else:
            return x * tf.exp(logs) + t
    def ld(x, logs):
        h, w = x.get_shape().as_list()[1:3]
        val = tf.reduce_sum(logs) * h * w
        if backward:
            return -val
        else:
            return val

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        t = tf.get_variable("t", (1, 1, 1, gs(x)[-1]), trainable=True)
        logs = tf.get_variable("logs", (1, 1, 1, gs(x)[-1]), trainable=True)
        if init:
            x_mean, x_var = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=True)
            logs_init = tf.log(1. / (tf.sqrt(x_var) + eps))
            t_init = -tf.exp(logs_init) * x_mean
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
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
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
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
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
    xs = squeeze(x)

    # z_init, _ = actnorm(x, "an", True, logdet=True)
    # z, logdet = actnorm(x, "an", False, logdet=True)
    # x_recons, logdet_recons = actnorm(z, "an", False, logdet=True, backward=True)


    # m = mask(xs, "checker", 0)
    # z_init, _ = coupling_layer(xs, m, "test_cl", init=True)
    # z, logdet = coupling_layer(xs, m, "test_cl")
    # z_init, logdet = coupling_block(xs, "b1", t="checker", init=True)
    # z, _ = coupling_block(xs, "b1", t="checker")

    # z_init, _ = scale_block(xs, "s1", True)
    # z, logdet = scale_block(xs, "s1")
    # xs_recons, logdet_recons = scale_block(z, "s1", backward=True)
    # x_recons = squeeze(xs_recons, backward=True)


    # 1 / 0

    # z_init, logdet = coupling_block(xs, "b1", t="channel", init=True)
    # z, _ = coupling_block(xs, "b1", t="channel")
    # z = tf.reshape(z, [-1, 16 * 16 * 12])
    #
    # z_init, _ = scale_block(xs, "s1", True)
    # z, logdet = scale_block(xs, "s1")
    # z = flatten(z)

    z_init, _ = net(xs, "net", 2, init=True)
    z, logdet = net(xs, "net", 2)
    xs_recons, logdet_recons = net_backwards(z, "net", init=False)
    x_recons = squeeze(xs_recons, backward=True)
    recons_error = tf.reduce_mean(tf.square(x - x_recons))

    logpz = tf.add_n([tf.reduce_sum(normal_logpdf(_z, 0., 0.), axis=[1, 2, 3]) for _z in z])
    logpz = tf.reduce_mean(logpz)
    logdet = tf.reduce_mean(logdet)
    total_logprob = logpz + logdet
    loss = -total_logprob / np.prod(gs(xs)[1:])
    lr = tf.placeholder(tf.float32, [], name="lr")
    optim = tf.train.AdamOptimizer(lr)
    opt = optim.minimize(loss)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("logdet", logdet)
    tf.summary.scalar("logp", logpz)
    tf.summary.scalar("recons", recons_error)

    # for v in tf.get_collection("logdets"):
    #     print(v.name)
    #     tf.summary.scalar(v.name, tf.reduce_mean(v))
    for v in tf.trainable_variables():
        tf.summary.histogram(v.name, v)
    sum_op = tf.summary.merge_all()
    #
    # #LOGDET IS FUXED YO
    #
    # # nn = NN(x, "test")
    # # print(gs(x), gs(nn))
    # #
    # # m = mask(x, "checker", 1)
    # # print(m[0, :, :, 0])
    # #
    # # cl, ld = coupling_layer(x, mask(x, "checker", 0), "test_cl")
    # # print(gs(cl))
    # #
    # # s = squeeze(x)
    # # print(gs(x), gs(s))
    #
    # #

    sess.run(tf.global_variables_initializer())
    sess.run(z_init, feed_dict={x: init_batch})
    #_an, _ld = sess.run([z, logdet], feed_dict={x: init_batch})
    # print(_an.min(), _an.max())
    # print(_an.mean(), _an.var())
    # print(_ld)

    writer = tf.summary.FileWriter("/tmp/train")

    for i in range(10000):
        batch_inds = np.random.choice(inds, 64, replace=False)
        batch = data_pad[batch_inds]
        iter_lr = .0003 # if i > 1000 else (i / 1000.) * .0003
        _l, logp, lgdt, _, sstr, _z = sess.run([loss, logpz, logdet, opt, sum_op, z], feed_dict={x: batch, lr: iter_lr})
        writer.add_summary(sstr, i)

        # if i % 10 == 0:
        #     batch_inds = np.random.choice(test_inds, 64, replace=False)
        #     test_batch = data_test_pad[batch_inds]
        #     logpt = sess.run(logpz, feed_dict={x: test_batch})
        #     print(i, logp, logpt)
        # else:
        print(i, _l, logp, lgdt)
        #print(_z[0])

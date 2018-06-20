import tensorflow as tf
import numpy as np
import utils
from utils import gs
import argparse
import os
import shutil
import json
import time

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


def coupling_layer(in_feats, name, init=False, backward=False, eps=1e-6, dim=32):
    def get_vars(feats):
        logit_s = NN(feats, "logit_s", init=init, dim=dim) + 2.
        s = tf.sigmoid(logit_s) + eps
        t = NN(feats, "t", init=init, dim=dim)
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


def flow_step(in_feats, name, init=False, backward=False, dim=32):
    with tf.variable_scope(name, reuse=(not init)):
        if backward:
            y = in_feats
            x, logdet_coupling = coupling_layer(y, "coupling", init=init, backward=True, dim=dim)
            x, logdet_invconv = invconv_layer(x, "invconv", init=init, backward=True)
            x, logdet_actnorm = actnorm(x, "actnorm", init=init, return_logdet=True, backward=True)
            return x, logdet_actnorm + logdet_invconv + logdet_coupling
        else:
            x = in_feats
            x, logdet_actnorm = actnorm(x, "actnorm", init=init, return_logdet=True, backward=False)
            x, logdet_invconv = invconv_layer(x, "invconv", init=init, backward=False)
            y, logdet_coupling = coupling_layer(x, "coupling", init=init, backward=False, dim=dim)
            return y, logdet_actnorm + logdet_invconv + logdet_coupling


def scale_block(in_feats, name, flow_steps, init=False, backward=False, dim=32):
    with tf.variable_scope(name, reuse=(not init)):
        if backward:
            z = in_feats
            logdet = 0.
            for i in reversed(range(flow_steps)):
                z, step_logdet = flow_step(z, "step{}".format(i), init=init, backward=True, dim=dim)
                logdet += step_logdet
            x = squeeze(z, backward=True)
            return x, logdet
        else:
            x = in_feats
            z = squeeze(x, backward=False)
            logdet = 0.
            for i in range(flow_steps):
                z, step_logdet = flow_step(z, "step{}".format(i), init=init, backward=False, dim=dim)
                logdet += step_logdet
            return z, logdet


def net(in_feats, name, num_blocks, flow_steps, init=False, backward=False, width=32):
    with tf.variable_scope(name, reuse=(not init)):
        if backward:
            zs = in_feats
            logdet = 0.
            z = zs[-1]
            for i in reversed(range(num_blocks)):
                z, block_logdet = scale_block(z, "block{}".format(i), flow_steps, init=init, backward=True, dim=width)
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
                z, block_logdet = scale_block(z, "block{}".format(i), flow_steps, init=init, backward=False, dim=width)
                logdet += block_logdet
                if i == num_blocks - 1:
                    zs.append(z)
                else:
                    z, zi = split(z)
                    zs.append(zi)
            return zs, logdet


def logpx(zs, logdet):
    logpz = tf.add_n([tf.reduce_sum(utils.normal_logpdf(z, 0., 0.), axis=[1, 2, 3]) for z in zs])
    ave_logpz = tf.reduce_mean(logpz)
    ave_logdet = tf.reduce_mean(logdet)
    total_logprob = (ave_logpz + ave_logdet)
    tf.summary.scalar("logdet", ave_logdet)
    tf.summary.scalar("logp", ave_logpz)
    return total_logprob


def create_experiment_directory(args):
    # write params
    with open(os.path.join(args.train_dir, "params.txt"), 'w') as f:
        f.write(json.dumps(args.__dict__))
    # copy code
    code_dest_dir = os.path.join(args.train_dir, "code")
    os.mkdir(code_dest_dir)
    code_dir = os.path.dirname(__file__)
    code_dir = '.' if code_dir == '' else code_dir
    python_files = [os.path.join(code_dir, fn) for fn in os.listdir(code_dir) if fn.endswith(".py")]
    for pyf in python_files:
        print(pyf, code_dest_dir)
        shutil.copy2(pyf, code_dest_dir)
    os.mkdir(os.path.join(args.train_dir, "best"))
    os.mkdir(os.path.join(args.train_dir, "backup"))


def get_lr(epoch, args):
    epoch_lr = (args.lr * (epoch + 1) / args.epochs_warmup) if epoch < args.epochs_warmup + 1 else args.lr
    # get decayed lr
    if args.lr_scalemode == 0:
        return epoch_lr
    else:
        lr_scale = args.decay_factor ** (epoch // args.epochs_decay)
        epoch_lr *= lr_scale
        return epoch_lr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/tmp/train")
    parser.add_argument("--dataset", type=str, default='mnist', help="Problem (mnist/cifar10/svhn)")
    parser.add_argument("--num_valid", type=int, default=None,
                        help="The number of examples to place into the validaiton set (only for svhn and cifar10)")
    parser.add_argument("--load_path", type=str, default=None, help="Path for load saved checkpoint from")
    parser.add_argument("--log_iters", type=int, default=1000, help="iters per each print and summary")

    # Optimization hyperparams:
    parser.add_argument("--epochs", type=int, default=100000, help="Train epoch size")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--init_batch_size", type=int, default=1024, help="batch size for init")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--lr_scalemode", type=int, default=0, help="Type of learning rate scaling. 0=none, 1=step.")
    parser.add_argument("--epochs_warmup", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--epochs_valid", type=int, default=1, help="Epochs between valid")
    parser.add_argument("--epochs_backup", type=int, default=10, help="Epochs between backup saving")
    parser.add_argument("--epochs_decay", type=int, default=250, help="Epochs between lr decay")
    parser.add_argument("--decay_factor", type=float, default=.1, help="Multiplier on learning rate")

    # Model hyperparams:
    parser.add_argument("--width", type=int, default=128, help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=8, help="Depth of network")
    parser.add_argument("--n_levels", type=int, default=3, help="Number of levels")
    parser.add_argument("--n_bits_x", type=int, default=5, help="Number of bits of x")

    # Finetuning arguments
    parser.add_argument("--finetune", type=int, default=0, help="if 0, then train generaitve, 1 then finetune")
    parser.add_argument("--clf_type", type=str, default="unwrap")

    args = parser.parse_args()
    args.n_bins_x = 2.**args.n_bits_x
    assert args.finetune in (0, 1)
    assert args.clf_type in ("unwrap", "pool")
    assert not os.path.exists(args.train_dir), "This directory already exists..."
    train_writer = tf.summary.FileWriter(os.path.join(args.train_dir, "train"))
    test_writer = tf.summary.FileWriter(os.path.join(args.train_dir, "test"))
    valid_writer = tf.summary.FileWriter(os.path.join(args.train_dir, "valid"))
    # setup experiment directory, copy current version of the code, save parameters
    create_experiment_directory(args)

    # create session
    sess = tf.Session()

    dataset_fn = {'svhn': utils.SVHNDataset, 'cifar10': utils.CIFAR10Dataset, 'mnist': utils.MNISTDataset}[args.dataset]
    dataset = dataset_fn(
        args.batch_size, init_size=args.init_batch_size, n_valid=args.num_valid, n_bits_x=args.n_bits_x
    )

    # unpack labeled examples
    x, y = dataset.x, dataset.y

    # build graph
    z_init, _ = net(x, "net", args.n_levels, args.depth, width=args.width, init=True)
    z, logdet = net(x, "net", args.n_levels, args.depth, width=args.width)

    # train to optimize logp(x)
    if args.finetune == 0:
        # input reconstructions
        x_recons, _ = net(z, "net", args.n_levels, args.depth, width=args.width, backward=True)
        # samples
        z_samp = [tf.random_normal(tf.shape(_z)) for _z in z]
        x_samp, _ = net(z_samp, "net", args.n_levels, args.depth, width=args.width, backward=True)

        # compute objective logp(x)
        logpz = tf.add_n([tf.reduce_sum(utils.normal_logpdf(z_l, 0., 0.), axis=[1, 2, 3]) for z_l in z])
        logpx = logpz + logdet
        objective = tf.reduce_mean(logpx) - np.log(args.n_bins_x) * np.prod(gs(x)[1:])
        loss = -objective

        # create optimizer
        lr = tf.placeholder(tf.float32, [], name="lr")
        optim = tf.train.AdamOptimizer(lr)
        opt = optim.minimize(loss)

        # summaries and visualizations
        x_recons = utils.postprocess(x_recons, n_bits_x=args.n_bits_x)
        x_samp = utils.postprocess(x_samp, n_bits_x=args.n_bits_x)
        recons_error = tf.reduce_mean(tf.square(tf.to_float(utils.postprocess(x, n_bits_x=args.n_bits_x) - x_recons)))
        tf.summary.image("x_sample", x_samp)
        tf.summary.image("x_recons", x_recons)
        tf.summary.image("x", x)
        tf.summary.scalar("recons", recons_error)
        tf.summary.scalar("lr", lr)
        # summaries for test datasets
        loss_summary = tf.summary.scalar("loss", loss)
        m_logdet = tf.reduce_mean(logdet)
        m_logpz = tf.reduce_mean(logpz)
        logdet_summary = tf.summary.scalar("logdet", m_logdet)
        logpz_summary = tf.summary.scalar("logpz", m_logpz)
        # the summary op to call for the training data
        train_summary = tf.summary.merge_all()
        # get all summaries we want to display on the test set
        test_summaries = [loss_summary, logdet_summary, logpz_summary]
        # get the values we need to call to get mean stats
        test_values = [loss, m_logdet, m_logpz]
        test_value_names = ['loss', 'logdet', 'logpz']
        test_summary = tf.summary.merge(test_summaries)

        # initialize variables
        sess.run(tf.global_variables_initializer())
        # initialize act-norm parameters
        sess.run(dataset.use_init)
        sess.run(z_init)

        # create savers for the best validation models and regularly
        best_saver = tf.train.Saver(max_to_keep=5)
        backup_saver = tf.train.Saver(max_to_keep=5)

        # restore model if asked
        if args.load_path is not None:
            backup_saver.restore(sess, args.load_path)
            start_epoch = int(args.load_path.split('-')[-1])
        else:
            start_epoch = 0

        # evaluation code block
        def evaluate(init_op, writer, name):
            sess.run(init_op)
            summary_values = []
            while True:
                try:
                    summary_values.append(sess.run(test_values))
                except tf.errors.OutOfRangeError:
                    summary_values = np.array(summary_values).mean(axis=0)
                    print("{}: ...".format(name))
                    for val_name, val_val in zip(test_value_names, summary_values):
                        print("    {}: {}".format(val_name, val_val))
                    fd = {node: val for node, val in zip(test_values, summary_values)}
                    sstr = sess.run(test_summary, feed_dict=fd)
                    writer.add_summary(sstr, cur_iter)
                    # return accuracy to determine best model
                    return summary_values[0]

        # training loop
        cur_iter = 0
        best_valid = np.inf
        for epoch in range(start_epoch, args.epochs):
            sess.run(dataset.use_train)
            t_start = time.time()
            # get lr for this epoch
            epoch_lr = get_lr(epoch, args)
            while True:
                try:
                    if cur_iter % args.log_iters == 0:
                        _re, _l, _, sstr = sess.run([recons_error, loss, opt, train_summary], feed_dict={lr: epoch_lr})
                        train_writer.add_summary(sstr, cur_iter)
                        print(cur_iter, _l, _re)

                    else:
                        _ = sess.run(opt, feed_dict={lr: epoch_lr})

                    cur_iter += 1
                except tf.errors.OutOfRangeError:
                    print("Completed epoch {} in {}".format(epoch, time.time() - t_start))
                    break

            # get accuracy on test set
            if epoch % args.epochs_valid == 0:
                evaluate(dataset.use_test, test_writer, "Test")
                # if we have a validation set, get validation accuracy
                if dataset.use_valid is not None:
                    valid_loss = evaluate(dataset.use_valid, valid_writer, "Valid")
                    if valid_loss < best_valid:
                        print("Best performing model with loss: {}".format(valid_loss))
                        best_valid = valid_loss
                        best_saver.save(sess, "{}/best/model.ckpt".format(args.train_dir), global_step=epoch)

            # backup model
            if epoch % args.epochs_backup == 0:
                backup_saver.save(sess, "{}/backup/model.ckpt".format(args.train_dir), global_step=epoch)

    # finetune classification
    else:
        # initialize variables
        sess.run(tf.global_variables_initializer())
        # initialize act-norm parameters
        sess.run(dataset.use_init)
        sess.run(z_init)

        # create savers for the best validation models and regularly
        best_saver = tf.train.Saver(max_to_keep=5)
        backup_saver = tf.train.Saver(max_to_keep=5)
        # restore pretrained weights
        if args.load_path is not None:
            backup_saver.restore(sess, args.load_path)

        # get top z as features for classification layer
        top_z = z[-1]
        if args.clf_type == "unwrap":
            zsize = np.prod(gs(top_z)[1:])
            feats = tf.reshape(top_z, [-1, zsize])
        elif args.clf_type == "pool":
            feats = tf.reduce_mean(top_z, axis=[1, 2])

        logits = tf.layers.dense(feats, dataset.n_class, name="logits")
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
        preds = tf.argmax(logits, axis=1)
        correct = tf.to_float(tf.equal(preds, y))
        accuracy = tf.reduce_mean(correct)






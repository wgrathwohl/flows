from glow import *
import argparse
import os
import time
import json
import shutil


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="/tmp/train")
    parser.add_argument("--dataset", type=str, default='mnist', help="Problem (mnist/cifar10/svhn)")
    parser.add_argument("--num_valid", type=int, default=None,
                        help="The number of examples to place into the validaiton set (only for svhn and cifar10)")
    parser.add_argument("--num_labels", type=int, default=None, help="Number of labeled examples to use")

    # Optimization hyperparams:
    parser.add_argument("--epochs", type=int, default=50000, help="Train epoch size")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--init_batch_size", type=int, default=1024, help="batch size for init")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--lr_scalemode", type=int, default=0, help="Type of learning rate scaling. 0=none, 1=linear, 2=sqrt.")
    parser.add_argument("--epochs_warmup", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--epochs_valid", type=int, default=10, help="Epochs between valid")

    # Model hyperparams:
    parser.add_argument("--width", type=int, default=64, help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=4, help="Depth of network")
    parser.add_argument("--n_levels", type=int, default=3, help="Number of levels")
    parser.add_argument("--disc_weight", type=float, default=0.00, help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=5, help="Number of bits of x")

    args = parser.parse_args()
    args.n_bins_x = 2.**args.n_bits_x
    assert 0. <= args.disc_weight <= 1., "Disc weigt must be in [0., 1.]"
    # set up logging
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
        args.batch_size, init_size=args.init_batch_size, n_labels=args.num_labels, n_valid=args.num_valid
    )

    # unpack labeled examples
    x, y = dataset.x, dataset.y
    y = tf.to_int64(y)
    y_oh = tf.one_hot(y, dataset.n_class)
    xpp = preprocess(x, n_bits_x=args.n_bits_x)

    z_init, _ = net(xpp, "net", args.n_levels, args.depth, width=args.width, init=True)
    z, logdet = net(xpp, "net", args.n_levels, args.depth, width=args.width)
    # i should remove this maybe to save a little time?
    x_recons, _ = net(z, "net", args.n_levels, args.depth, width=args.width, backward=True)
    # make parameters for mixture centers
    top_z_shape = gs(z[-1])[1:]
    class_mu = tf.get_variable(
        "class_mu", [dataset.n_class] + top_z_shape,
        dtype=tf.float32, initializer=tf.random_normal_initializer(),
        trainable=True
    )
    # sample from N(0, I) for low level features and MOG for top level features
    z_samp = [tf.random_normal(tf.shape(_z)) for _z in z[:-1]] + [utils.mog_sample(class_mu, tf.shape(z[-1]))]
    x_samp, _ = net(z_samp, "net", args.n_levels, args.depth, width=args.width, backward=True)

    # get means for top features for mini batch elements
    mu_z_top = tf.reduce_sum(y_oh[:, :, None, None, None] * class_mu[None, :, :, :, :], axis=1)
    mu_z = ([0.] * (len(z) - 1)) + [mu_z_top]

    # compute generative objective logp(x, y)
    logpy = np.log(1. / dataset.n_class)
    lp = lambda z_l, mu_l: tf.reduce_sum(utils.normal_logpdf(z_l, mu_l, 0.), axis=[1, 2, 3])
    logpz_given_y = tf.add_n([lp(mu_l, z_l) for mu_l, z_l in zip(mu_z, z)])
    logpx_given_y = logpz_given_y + logdet
    logpxy = logpx_given_y + logpy
    gen_objective_l = tf.reduce_mean(logpxy) - np.log(args.n_bins_x) * np.prod(gs(x)[1:])
    # compute discriminative objective logp(y|x)
    top_z = z[-1]
    logpz_given_y_all = tf.reduce_sum(
        utils.normal_logpdf(top_z[:, None, :, :, :], class_mu[None, :, :, :, :], 0.),
        axis=[2, 3, 4]
    )
    logpy_given_z = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logpz_given_y_all)
    disc_objective = tf.reduce_mean(logpy_given_z)
    preds = tf.argmax(logpz_given_y_all, axis=1)
    correct = tf.to_float(tf.equal(preds, y))
    accuracy = tf.reduce_mean(correct)

    # compute (or fake) generaitve objective for unlabeled data
    if args.num_labels is None:
        gen_objective_u = 0.
        l_weight = 1.
        u_weight = 0.
    else:
        x_u = dataset.x_u
        xpp_u = preprocess(x_u, n_bits_x=args.n_bits_x)
        z_u, logdet_u = net(xpp_u, "net", args.n_levels, args.depth, width=args.width)
        # get logp(z) for intermediate z since they don't depend on y
        logpz_mid_u = tf.add_n([lp(0., z_u_l) for z_u_l in z_u[:-1]])
        # get logp(z) for top z by taking logsumexp of each logp(z|y)
        logpz_given_y_all_u = tf.reduce_sum(
            utils.normal_logpdf(z_u[-1][:, None, :, :, :], class_mu[None, :, :, :, :], 0.),
            axis=[2, 3, 4]
        )
        logpz_top_u = tf.reduce_logsumexp(logpz_given_y_all_u, axis=1)
        logpz_u = logpz_mid_u + logpz_top_u + logpy
        logpx_u = logpz_u + logdet_u
        gen_objective_u = tf.reduce_mean(logpx_u) - np.log(args.n_bins_x) * np.prod(gs(x)[1:])
        tf.summary.scalar("gen_objective_u", gen_objective_u)
        n_train = float(dataset.n_train_l + dataset.n_train_u)
        l_weight = dataset.n_train_l / n_train
        u_weight = dataset.n_train_u / n_train

    # total generative objective
    gen_objective = l_weight * gen_objective_l + u_weight * gen_objective_u
    # total objective
    objective = args.disc_weight * disc_objective + (1. - args.disc_weight) * gen_objective
    loss = -objective
    lr = tf.placeholder(tf.float32, [], name="lr")
    optim = tf.train.AdamOptimizer(lr)
    opt = optim.minimize(loss)

    # summaries and visualizations
    x_recons = postprocess(x_recons, n_bits_x=args.n_bits_x)
    x_samp = postprocess(x_samp, n_bits_x=args.n_bits_x)
    recons_error = tf.reduce_mean(tf.square(tf.to_float(postprocess(xpp, n_bits_x=args.n_bits_x) - x_recons)))
    tf.summary.image("x_sample", x_samp)
    tf.summary.image("x_recons", x_recons)
    tf.summary.image("x", x)
    tf.summary.scalar("recons", recons_error)
    tf.summary.scalar("lr", lr)
    loss_summary = tf.summary.scalar("loss", loss)
    acc_summary = tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("logdet", tf.reduce_mean(logdet))
    tf.summary.scalar("logpz", tf.reduce_mean(logpz_given_y))
    tf.summary.scalar("disc_objective", disc_objective)
    tf.summary.scalar("gen_objective_l", gen_objective_l)
    train_summary = tf.summary.merge_all()
    #test_summary = tf.summary.merge([loss_summary, acc_summary])
    test_summary = acc_summary

    # initialize variables
    sess.run(tf.global_variables_initializer())
    # initialize act-norm parameters
    sess.run(dataset.use_init)
    sess.run(z_init)

    # evaluation code block
    def evaluate(init_op, writer, name):
        sess.run(init_op)
        test_acc = []
        while True:
            try:
                _c = sess.run(correct)
                test_acc.extend(_c)
            except tf.errors.OutOfRangeError:
                # at epoch end
                test_acc = np.mean(test_acc)
                print("{} acc: {}".format(name, test_acc))
                sstr = sess.run(test_summary, feed_dict={accuracy: test_acc})
                writer.add_summary(sstr, cur_iter)
                break


    # training loop
    cur_iter = 0
    for epoch in range(args.epochs):
        sess.run(dataset.use_train)
        t_start = time.time()
        while True:
            try:
                cur_lr = (args.lr * (epoch + 1) / args.epochs_warmup) if epoch < args.epochs_warmup + 1 else args.lr
                if cur_iter % 100 == 0:
                    _re, _l, _a, _, sstr = sess.run([recons_error, loss, accuracy, opt, train_summary],
                                                    feed_dict={lr: cur_lr})
                    train_writer.add_summary(sstr, cur_iter)
                    print(cur_iter, _l, _a, _re)

                else:
                    _ = sess.run(opt, feed_dict={lr: cur_lr})

                cur_iter += 1
            except tf.errors.OutOfRangeError:
                print("Completed epoch {} in {}".format(epoch, time.time() - t_start))
                break

        # get accuracy on test set
        if epoch % args.epochs_valid == 0:
            evaluate(dataset.use_test, test_writer, "Test")
            # if we have a validation set, get validation accuracy
            if dataset.use_valid is not None:
                evaluate(dataset.use_valid, valid_writer, "Valid")


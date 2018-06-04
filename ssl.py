from glow import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mnist', help="Problem (mnist/cifar10/svhn)")

    # Optimization hyperparams:
    parser.add_argument("--epochs", type=int, default=50000, help="Train epoch size")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--init_batch_size", type=int, default=1024, help="batch size for init")
    parser.add_argument("--lr", type=float, default=0.001, help="Base learning rate")
    parser.add_argument("--lr_scalemode", type=int, default=0, help="Type of learning rate scaling. 0=none, 1=linear, 2=sqrt.")
    parser.add_argument("--epochs_warmup", type=int, default=10, help="Warmup epochs")
    parser.add_argument("--epochs_valid", type=int, default=10, help="Epochs between valid")

    # Model hyperparams:
    parser.add_argument("--width", type=int, default=128, help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=4, help="Depth of network")
    parser.add_argument("--n_levels", type=int, default=3, help="Number of levels")
    parser.add_argument("--weight_y", type=float, default=0.00, help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=5, help="Number of bits of x")

    args = parser.parse_args()
    sess = tf.Session()

    dataset_fn = {'svhn': utils.SVHNDataset, 'cifar10': utils.CIFAR10Dataset, 'mnist': utils.MNISTDataset}[args.dataset]
    dataset = dataset_fn(args.batch_size, init_size=args.init_batch_size)

    x, y = dataset.iterator.get_next()
    y_oh = tf.one_hot(y, dataset.n_class)
    xpp = preprocess(x, n_bits_x=args.n_bits_x)

    z_init, _ = net(xpp, "net", args.n_levels, args.depth, width=args.width, init=True)
    z, logdet_orig = net(xpp, "net", args.n_levels, args.depth, width=args.width)
    x_recons, logdet_recons = net(z, "net", args.n_levels, args.depth, width=args.width, backward=True)
    # make parameters for mixture centers
    top_z_shape = gs(z[-1])[1:]
    class_mu = tf.get_variable(
        "class_mu", [dataset.n_class] + top_z_shape,
        dtype=tf.float32, initializer=tf.random_normal_initializer(),
        trainable=True
    )
    # sample from N(0, I) for low level features and MOG for top level features
    z_samp = [tf.random_normal(tf.shape(_z)) for _z in z[:-1]] + [utils.mog_sample(class_mu, tf.shape(z[-1]))]
    x_samp, logdet_samp = net(z_samp, "net", args.n_levels, args.depth, width=args.width, backward=True)

    # get means for top features for mini batch elements
    mu_z_top = tf.reduce_sum(y_oh[:, :, None, None, None] * class_mu[None, :, :, :, :], axis=1)

    # compute loss
    logpy = np.log(1. / dataset.n_class)
    logpx_given_y = None
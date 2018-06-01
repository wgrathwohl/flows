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
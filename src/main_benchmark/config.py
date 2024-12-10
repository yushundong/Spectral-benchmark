import argparse


def config():
    parser = argparse.ArgumentParser(description="Graph Node Regression with GNN")
    parser.add_argument("--checkpoint", type=str, help="gnn checkpoint")
    parser.add_argument(
        "--dataset",
        type=str,
        default="Cora",
        help='Graph dataset. Use "all" to run all datasets.',
    )
    parser.add_argument(
        "--gnn", type=str, default="GCN", help='GNN model. Use "all" to run all GNNs.'
    )
    parser.add_argument("--layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--batchnorm", action="store_true", help="Use BatchNorm")
    parser.add_argument(
        "--device",
        type=int,
        choices=[0, 1, 2, 3, 4, 5],
        default=0,
        help="CUDA device number",
    )
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate")
    parser.add_argument("--epochs", type=int, default=500, help="the number of epochs")
    parser.add_argument(
        "--h_dim", type=int, default=32, help="the dimension of latent space"
    )
    parser.add_argument(
        "--bin_width", type=float, default=0.1, help="bin width for spectral binning."
    )
    parser.add_argument(
        "--num_node_classes", type=int, default=4, help="number of node classes"
    )
    parser.add_argument(
        "--num_exp_times", type=int, default=3, help="number of times to run the experiment"
    )

    parser.add_argument(
        "--bins",
        nargs="+",
        default=[-1.0],
        help="list of bins to use as our e_i. leave blank if you don't want to use prediction by binning.",
    )

    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--layer_sizes", nargs="+", default=[100, 100])
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results_dir", default="results", help="use for downstream tasks")

    return parser.parse_args()

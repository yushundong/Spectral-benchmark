import torch
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List


@dataclass
class Args:
    dataset: str

    # NN args
    gnn_model: str
    layers: int
    h_dim: int
    dropout: float
    lr: float
    epochs: int
    batch_size: int
    gat_num_heads: int
    batchnorm: bool
    device: str
    train: bool

    seeds: List[int]
    tag: str

    # path args
    data_dir: str
    checkpoint: str

    train_val_test_ratios: List[float]
    target_frequency: str
    feature_frequency: str
    bin_width: int
    feat_dim: int
    mlp_layers: int
    frequency_cutoffs: List[float]
    num_node_classes: int


def get_args() -> Args:
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default="Cora")
    parser.add_argument("--gnn_model", type=str, default="GCN")
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--h_dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="0")

    parser.add_argument("--gat_num_heads", type=int, default=4)
    parser.add_argument("--batchnorm", action="store_true")

    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--tag", type=str, default="")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--train", action="store_true")

    parser.add_argument(
        "--frequency_cutoffs",
        nargs="+",
        type=float,
        default=[1.0],
    )
    parser.add_argument(
        "--train_val_test_ratios",
        nargs="+",
        type=float,
        default=[0.8, 0.1, 0.1],
    )
    parser.add_argument("--target_frequency", type=str, default="high")
    parser.add_argument("--feature_frequency", type=str, default="low")
    parser.add_argument("--bin_width", type=int, default=50)
    parser.add_argument(
        "--feat_dim",
        help="Dimension of the feature matrix. If you want to match the number of frequencies as the feature dimensino, use -1.",
        type=int,
    )

    parser.add_argument("--mlp_layers", type=int, default=3)
    parser.add_argument("--num_node_classes", type=int, default=1, help="1 for regression")

    args = parser.parse_args()
    args = Args(**vars(args))

    return args


args = get_args()

assert len(args.frequency_cutoffs) < 4
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

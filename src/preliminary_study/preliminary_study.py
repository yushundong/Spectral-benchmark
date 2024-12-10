import numpy as np
import torch
import os
import seaborn as sb
import matplotlib.pyplot as plt

from torch import Tensor
from typing import List, Tuple
from utils import GNNModel, MLP
from utils import get_laplacian_spectrum, dataset_picker, make_splits
from config import args, device

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["mathtext.fontset"] = (
    "stix"
)
plt.rcParams["mathtext.default"] = "regular"
sb.set_palette("tab10")
mse_loss = torch.nn.functional.mse_loss


def bin_eigenvectors(eigenvectors: Tensor, bin_width: float) -> Tensor:
    binned_eigenvectors = torch.zeros(eigenvectors.shape[0], bin_width).to(device)
    n = eigenvectors.shape[1]
    bin_size = n // bin_width

    for i in range(bin_width):
        start = i * bin_size
        end = (i + 1) * bin_size

        binned_eigenvectors[:, i] = eigenvectors[:, start:end].mean(dim=1)
        binned_eigenvectors[:, i] /= binned_eigenvectors[:, i].norm(p=2)

    return binned_eigenvectors


def get_energy_distribution(
    signal: Tensor, eigenvectors: Tensor, frequency_cutoffs=[1.0], bin_width=100
) -> Tuple[Tensor, List[Tensor]]:
    eigenvectors = bin_eigenvectors(eigenvectors, bin_width)

    n = eigenvectors.shape[1]
    energies = torch.zeros(n)
    for i in range(n):
        ip = torch.dot(signal, eigenvectors[:, i]) ** 2
        energies[i] = ip

    energies /= energies.sum()
    start = 0
    energies_per_interval = []

    for interval in frequency_cutoffs:
        indices = range(start, start + int(interval * n))
        energies_per_interval.append(energies[indices].sum().item())

        start = start + int(interval * n)

    return energies, energies_per_interval


def get_frequencies(
    eigenvectors: Tensor, frequency_cutoffs: List[float]
) -> List[Tensor]:
    n = eigenvectors.shape[1]
    mean_frequencies = []
    frequencies = []
    start = 0

    for interval in frequency_cutoffs:
        indices = range(start, start + int(interval * n))
        frequency = eigenvectors[:, indices].mean(dim=1)
        frequency /= frequency.norm(p=2)
        mean_frequencies.append(frequency)
        frequencies.append(
            eigenvectors[:, indices] / eigenvectors[:, indices].norm(p=2)
        )

        start += int(interval * n)

    return mean_frequencies, frequencies


def experiment():
    assert len(args.frequency_cutoffs) == 3  # low, med, high cutoffs must be defined

    dataset, _, _, _ = dataset_picker(args.dataset)

    data = dataset.data
    edge_index = data.edge_index
    edge_weight = data.edge_attr

    try:
        eigenvalues, eigenvectors = torch.load(f"spectra/{args.dataset}_spectrum.pt")
    except FileNotFoundError:
        eigenvalues, eigenvectors = get_laplacian_spectrum(edge_index, edge_weight)
        if not os.path.exists("spectra"):
            os.makedirs("spectra")
        torch.save((eigenvalues, eigenvectors), f"spectra/{args.dataset}_spectrum.pt")

    label_to_axis = {
        ("low", "high"): (0, 0),
        ("high", "low"): (0, 1),
    }
    frequency_labels = ["low", "med", "high"]

    freq_cutoffs = [[0.33, 0.33, 0.33]]
    mask_ratios = [[0.8, 0.1, 0.1]]

    n_freq_cutoffs = len(freq_cutoffs)
    n_mask_ratios = len(mask_ratios)

    for label in label_to_axis.keys():
        feature_frequency, target_frequency = label
        fig, axs = (
            plt.subplots(
                n_freq_cutoffs,
                n_mask_ratios,
                figsize=(0.73 * 6, 0.73 * 3.5),
                squeeze=False,
            )
            if n_freq_cutoffs > 1 and n_mask_ratios > 1
            else plt.subplots(1, 1, figsize=(15, 10))
        )

        dirname = "results"
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        for i in range(n_freq_cutoffs):
            for j in range(n_mask_ratios):
                current_freq_cutoffs = freq_cutoffs[i]
                current_mask_ratios = mask_ratios[j]

                make_splits(dataset, current_mask_ratios)
                mean_frequencies, frequencies = get_frequencies(
                    eigenvectors, current_freq_cutoffs
                )

                train_mask = dataset.train_mask
                test_mask = dataset.test_mask

                target_frequencies = mean_frequencies[
                    frequency_labels.index(target_frequency)
                ]
                feature_frequencies = mean_frequencies[
                    frequency_labels.index(feature_frequency)
                ]

                dataset.y = target_frequencies.to(device)

                if args.feat_dim == -1:
                    dataset.x = frequencies[
                        frequency_labels.index(feature_frequency)
                    ].to(device)
                else:
                    dataset.x = feature_frequencies.to(device)
                    dataset.x = torch.stack([dataset.x] * args.feat_dim, dim=1)

                dataset.x = (dataset.x - dataset.x.mean(dim=0)) / dataset.x.std(dim=0)

                save_name = f"{args.gnn_model}-{args.dataset}-{args.tag}-feat_{feature_frequency}-tgt_{target_frequency}-freq_{current_freq_cutoffs}-mask_{current_mask_ratios}-feat_dim_{args.feat_dim}"

                try:
                    model_dicts = torch.load(f"results/{save_name}.pt")
                except Exception as e:
                    print(e)
                    model_dicts = train(dataset, save_name)

                output_signal, output_signal_stdev = run_exp(model_dicts, dataset)

                train_mse = mse_loss(output_signal[train_mask], dataset.y[train_mask])
                test_mse = mse_loss(output_signal[test_mask], dataset.y[test_mask])

                eigenvectors = eigenvectors.to(device)

                output_signal = output_signal * dataset.y.std() + dataset.y.mean()

                output_energies, _ = get_energy_distribution(
                    output_signal[test_mask],
                    eigenvectors[test_mask],
                    current_freq_cutoffs,
                    args.bin_width,
                )
                target_energies, _ = get_energy_distribution(
                    dataset.y[test_mask],
                    eigenvectors[test_mask],
                    current_freq_cutoffs,
                    args.bin_width,
                )
                if args.feat_dim != 1:
                    feature_signal = dataset.x.mean(dim=1)
                else:
                    feature_signal = dataset.x.squeeze(1)
                feature_energies, _ = get_energy_distribution(
                    feature_signal[test_mask],
                    eigenvectors[test_mask],
                    current_freq_cutoffs,
                    args.bin_width,
                )

                r = list(range(args.bin_width))
                ax = axs[i, j] if n_freq_cutoffs > 1 and n_mask_ratios > 1 else axs
                ax.tick_params(axis="both", which="major", labelsize=50)
                ax.tick_params(axis="both", which="minor", labelsize=50)

                ax.bar(
                    r,
                    target_energies,
                    color=(100 / 256, 200 / 256, 170 / 256),
                    alpha=0.73,
                    edgecolor="black",
                    linewidth=4,
                    capsize=2,
                )
                ax.bar(
                    r,
                    feature_energies,
                    color=(230 / 256, 150 / 256, 59 / 256),
                    alpha=0.73,
                    edgecolor="black",
                    linewidth=4,
                    capsize=2,
                )
                ax.bar(
                    r,
                    output_energies,
                    color=(89 / 256, 150 / 256, 197 / 256),
                    alpha=0.73,
                    edgecolor="black",
                    linewidth=4,
                    capsize=2,
                )

                plt.gca().set_facecolor("#EEF0F2")
                plt.grid(True, linestyle="--", color="gray", alpha=0.5)
                plt.grid(axis="y", linestyle="--", alpha=0.7)
                plt.xticks(rotation=35)
                ax.set_xlabel("Frequency bin", fontsize=54)
                ax.set_ylabel("Energy", fontsize=54)
                results_dict = {
                    "train_mse": train_mse.item(),
                    "test_mse": test_mse.item(),
                    "output_signal": output_signal,
                    "output_signal_stdev": output_signal_stdev,
                    "target_energies": target_energies,
                    "feature_energies": feature_energies,
                    "output_energies": output_energies,
                }
                torch.save(results_dict, f"{dirname}/{save_name}.pt")

        plot_name = f"{dirname}/{args.gnn_model}-feat_{feature_frequency}-tgt_{target_frequency}-feat_dim_{args.feat_dim}.pdf"

        if not os.path.exists(dirname):
            os.makedirs(dirname)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(plot_name, bbox_inches="tight", format="pdf", dpi=600)
        print(f"\nSaved to {plot_name}")
        plt.close()


def run_exp(model_dicts, dataset) -> Tuple[Tensor, Tensor]:
    output_signals = []
    if args.feat_dim == -1:
        in_channels = dataset.x.shape[1]
    else:
        in_channels = args.feat_dim

    if args.gnn_model != "MLP":
        model = GNNModel(args, args.gnn_model, in_channels=in_channels).to(device)
    else:
        model = MLP(
            args,
            in_channels=in_channels,
            layer_widths=[args.h_dim] * args.mlp_layers,
            num_classes=1,
        ).to(device)

    for model_dict in model_dicts:
        model.load_state_dict(model_dict)
        model.eval()

        test_masks = dataset.test_mask.to(device)

        if test_masks.dim() != 2:
            test_masks = test_masks.view(-1, 1)

        num_splits = test_masks.shape[1]

        edge_index = dataset.edge_index.to(device)
        x = dataset.x.to(device)

        output_per_split = []

        for _ in range(num_splits):
            if args.gnn_model != "MLP":
                pred = model(x, edge_index).detach()
            else:
                pred = model(x).detach()

            output_per_split.append(pred)

        avg_out_per_split = torch.stack(output_per_split).mean(dim=0)
        output_signals.append(avg_out_per_split)

    output_signal = torch.stack(output_signals).mean(dim=0).squeeze(1)
    output_signal_stdev = torch.stack(output_signals).std(dim=0).squeeze(1)

    return output_signal, output_signal_stdev


def train(dataset, save_name: str) -> List[dict]:
    models = []

    train_masks, val_masks, test_masks = (
        dataset.train_mask.to(device),
        dataset.val_mask.to(device),
        dataset.test_mask.to(device),
    )

    if train_masks.dim() != 2:
        train_masks = train_masks.view(-1, 1)
        val_masks = val_masks.view(-1, 1)
        test_masks = test_masks.view(-1, 1)

    num_splits = train_masks.shape[1]

    for split in range(num_splits):
        print(f"Split {split}")
        train_mask, val_mask, test_mask = (
            train_masks[:, split],
            val_masks[:, split],
            test_masks[:, split],
        )

        for seed in args.seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

            if args.feat_dim == -1:
                in_channels = dataset.x.shape[1]
            else:
                in_channels = args.feat_dim

            if args.gnn_model != "MLP":
                model = GNNModel(args, args.gnn_model, in_channels=in_channels).to(
                    device
                )
            else:
                model = MLP(
                    args,
                    in_channels=in_channels,
                    layer_widths=[args.h_dim] * args.mlp_layers,
                    num_classes=1,
                ).to(device)

            if args.checkpoint:
                model.load_state_dict(torch.load(args.checkpoint))

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=1
            )

            dir_name = "results"

            if not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)

            best_val_mse = float("inf")
            best_val_epoch = 0

            edge_index = dataset.edge_index.to(device)
            x = dataset.x.to(device)
            y = dataset.y.to(device)

            centered_y = (y - y.mean()) / y.std()

            for epoch in range(args.epochs):
                model.train()
                train_mse = 0
                optimizer.zero_grad()

                if args.gnn_model != "MLP":
                    pred = model(x, edge_index)[train_mask].squeeze(1)
                else:
                    pred = model(x)[train_mask].squeeze(1)

                loss = mse_loss(pred, centered_y[train_mask])
                loss.backward()
                optimizer.step()

                train_mse += loss.item()

                model.eval()

                with torch.no_grad():
                    if args.gnn_model != "MLP":
                        pred = model(x, edge_index).squeeze(1)
                    else:
                        pred = model(x).squeeze(1)

                    pred = pred * y.std() + y.mean()

                    val_mse = mse_loss(pred[val_mask], y[val_mask]).item()
                    test_mse = mse_loss(pred[test_mask], y[test_mask]).item()

                    if val_mse < best_val_mse:
                        print(
                            f"Epoch: {epoch}, New best val mse: {val_mse}, corresponding best test mse: {test_mse}",
                            end="\r",
                        )
                        best_val_mse = val_mse
                        best_val_epoch = epoch
                        best_model = model.state_dict()

                lr_scheduler.step()

            models.append(best_model)

    print(f"Achieved best val mse: {best_val_mse} at epoch {best_val_epoch}")

    torch.save(
        models,
        f"{dir_name}/{save_name}.pt",
    )

    return models

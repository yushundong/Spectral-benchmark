import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
import os

from dataclasses import dataclass
from config import config

from utils import (
    get_laplacian_spectrum,
    dataset_picker,
    GNNModel,
    gnn_models,
)


def transform_tensor(num_node_classes, tensor):
    cls_tensor = tensor.clone()
    gap = 2 / num_node_classes
    for i in range(num_node_classes):
        cls_tensor[(tensor >= -1 + i * gap) & (tensor < -1 + (i + 1) * gap)] = i

    cls_tensor = cls_tensor.long()
    counts = torch.zeros(num_node_classes, tensor.shape[1])
    for i in range(num_node_classes):
        counts[i] = torch.sum(cls_tensor == i, dim=0)

    return counts, cls_tensor


def get_cls_gts(num_node_classes, bin_width, eigenvalues, eigenvectors):
    bin_x = np.arange(bin_width / 2, 2, bin_width)
    bin_x = np.round(bin_x, decimals=2)
    ev_bins = [[] for i in range(int(2 / bin_width))]
    for i in range(len(eigenvalues)):
        bin_number = int(torch.div(eigenvalues[i], bin_width, rounding_mode="trunc"))
        if bin_number == int(2 / bin_width):
            bin_number -= 1
        ev_bins[bin_number].append(eigenvectors[:, i])
    idx_to_delete = []
    gts = []
    for i in range(len(ev_bins)):
        ev_bin = ev_bins[i]
        if not ev_bin:
            idx_to_delete.append(i)
        else:
            stacked_tensor = torch.stack(ev_bin)
            mean_tensor = torch.mean(stacked_tensor, dim=0)
            gts.append(mean_tensor)
    try:
        valid_bin_x = np.delete(bin_x, idx_to_delete)
    except:
        pass
    gts = torch.transpose(torch.stack(gts), 0, 1)
    counts, cls_gts = transform_tensor(num_node_classes, gts)
    return counts, valid_bin_x, cls_gts, gts


def condense_eigvecs(args, eigenvalues, eigenvectors):
    valid_indices = (eigenvalues >= 1e-3).nonzero().flatten()

    random_elements = torch.randperm(
        valid_indices.shape[0], generator=torch.Generator().manual_seed(42)
    )

    valid_indices = valid_indices[random_elements]

    eigenvalues = eigenvalues[valid_indices]
    eigenvectors = eigenvectors[:, valid_indices]

    counts, bin_x, cls_gts, gts = get_cls_gts(
        args.num_node_classes, args.bin_width, eigenvalues, eigenvectors
    )

    return counts, bin_x, cls_gts, gts


def get_random_indices(length: int, seed: int = 123):
    st0 = np.random.get_state()
    np.random.seed(seed)
    random_indices = np.random.permutation(length)
    np.random.set_state(st0)
    return random_indices


def train_loop(dataset, gnn, args, bin_idx=-1):
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    dataset_name = dataset
    dataset, train_mask, val_mask, test_mask = dataset_picker(dataset_name)
    data = dataset[0]

    data = data.to(device)
    edge_index = data.edge_index
    edge_weight = data.edge_attr

    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    num_nodes = data.num_nodes

    dataset_path = f"../../data/{dataset_name}/"

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    try:
        eigenvalues, eigenvectors = torch.load(f"{dataset_path}/spectrum.pt")
    except FileNotFoundError:
        eigenvalues, eigenvectors = get_laplacian_spectrum(edge_index, edge_weight)
        torch.save((eigenvalues, eigenvectors), f"{dataset_path}/spectrum.pt")

    bin_width = args.bin_width
    counts, bin_x, cls_gts, gts = get_cls_gts(
        args.num_node_classes, bin_width, eigenvalues, eigenvectors
    )
    use_bin_prediction = bin_idx != -1

    if use_bin_prediction:
        spectrum_point = int(gts.shape[1] * float(bin_idx))
        gts_x = gts[:, spectrum_point].reshape(-1, 1).to(device)
    else:
        spectrum_point = None
        gts_x = None

    torch.save(counts, dataset_path + "class_stats.pt")
    pred_acc_tensor = torch.zeros(args.num_exp_times, len(bin_x))
    predictions_all = [[] for _ in range(args.num_exp_times)]

    if use_bin_prediction:
        num_features = 1
    else:
        num_features = data.num_features

    for exp_num in range(args.num_exp_times):
        for i in tqdm.tqdm(range(len(bin_x))):
            model = GNNModel(
                args,
                gnn,
                num_features,
            ).to(device)

            if dataset == "Cycle":
                if torch.cuda.is_available():
                    train_mask, val_mask, test_mask = (
                        torch.tensor([]).to(args.device),
                        torch.tensor([]).to(args.device),
                        torch.tensor([]).to(args.device),
                    )
                else:
                    train_mask, val_mask, test_mask = (
                        torch.tensor([]),
                        torch.tensor([]),
                        torch.tensor([]),
                    )

                for cls in range(args.num_node_classes):
                    cls_inds = torch.where(cls_gts[:, i] == cls)[0]
                    if len(cls_inds) > 0:
                        cls_inds = cls_inds[torch.randperm(len(cls_inds))]
                        inds_length = len(cls_inds)
                        train_to_add = cls_inds[: int(0.6 * inds_length)]
                        val_to_add = cls_inds[
                            int(0.6 * inds_length) : int(0.8 * inds_length)
                        ]
                        test_to_add = cls_inds[
                            int(0.8 * inds_length) : int(1 * inds_length)
                        ]
                        train_mask = torch.cat((train_mask, train_to_add), 0)
                        val_mask = torch.cat((val_mask, val_to_add), 0)
                        test_mask = torch.cat((test_mask, test_to_add), 0)
                all_nodes = (
                    torch.arange(num_nodes).to(args.device)
                    if torch.cuda.is_available()
                    else torch.arange(num_nodes)
                )
                train_mask, val_mask, test_mask = (
                    torch.isin(all_nodes, train_mask.to(torch.long)),
                    torch.isin(all_nodes, val_mask.to(torch.long)),
                    torch.isin(all_nodes, test_mask.to(torch.long)),
                )

            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = torch.nn.CrossEntropyLoss()

            best_val = 10e9
            for epoch in range(args.epochs):
                optimizer.zero_grad()
                if use_bin_prediction:
                    output = model(gts_x, data.edge_index)
                else:
                    output = model(data.x, data.edge_index)

                cls_gts = cls_gts.to(device)
                loss = criterion(output[train_mask], cls_gts[:, i][train_mask])
                loss.backward()
                optimizer.step()

                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        if use_bin_prediction:
                            predictions = model(gts_x, data.edge_index)
                        else:
                            predictions = model(data.x, data.edge_index)
                        pred_ce = (
                            criterion(predictions[val_mask], cls_gts[:, i][val_mask])
                            * 10e6
                        )
                    if pred_ce < best_val:
                        best_model = model.state_dict()

            # Evaluate the model
            model = GNNModel(
                args,
                gnn,
                num_features,
            ).to(device)
            model.load_state_dict(best_model)
            model.eval()
            with torch.no_grad():
                if use_bin_prediction:
                    predictions = model(gts_x, data.edge_index)
                else:
                    predictions = model(data.x, data.edge_index)
                predictions = torch.argmax(predictions[test_mask], dim=1)
                acc = sum(predictions == cls_gts[:, i][test_mask]) / sum(test_mask)
                pred_acc_tensor[exp_num][i] = acc.item()
                predictions_all[exp_num].append(predictions)

            save_dir = f"./weights/{dataset_name}/{gnn}"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save(model.state_dict(), f"{save_dir}/{dataset}_bin_{i}_{gnn}.pt")

    pred, var = torch.mean(pred_acc_tensor, dim=0), torch.var(pred_acc_tensor, dim=0)
    plt.tight_layout()
    return (
        bin_x,
        pred,
        bin_width,
        var,
        dataset_name,
        spectrum_point,
        bin_idx,
        predictions_all,
    )


def save_results(args, items, gnn):
    model_str = gnn
    hyper_param_str = f"lr_{args.lr}_epochs_{args.epochs}_hdim_{args.h_dim}_layers_{args.layers}_dout_{args.dropout}_binwidith_{args.bin_width}"
    prefix = f"{args.results_dir}/{args.dataset}/{model_str}_{hyper_param_str}"

    pred_fname = f"{prefix}.pt"

    if not os.path.exists(os.path.dirname(pred_fname)):
        os.makedirs(os.path.dirname(pred_fname))

    print("Saved to: ", pred_fname)
    torch.save((items[0][1], items[0][3]), pred_fname)


def main(args):
    datasets = [args.dataset]
    gnns = [args.gnn] if args.gnn != "all" else list(gnn_models.keys())

    for gnn in gnns:
        items = []
        print(f"Training {gnn}...")

        for dataset in datasets:
            print(f"\tTraining on dataset: {dataset}...")

            for bin_idx in args.bins:
                train_items = train_loop(dataset, gnn, args, bin_idx)
                items.append(train_items)

        save_results(args, items, gnn)


if __name__ == "__main__":
    args = config()
    main(args)

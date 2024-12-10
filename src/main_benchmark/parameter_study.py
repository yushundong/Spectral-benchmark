import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

path_root = "./results"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.default"] = "regular"

models = ["Transformer", "SAGE", "GATv2", "GAT", "SGC"]
datasets = ["Photo", "DBLP", "CS", "Cora_Full"]
layer_counts = [2, 3, 4]
h_dim = 64
num_bins = 20

start_color = np.array([230 / 256, 150 / 256, 59 / 256])
end_color = np.array([89 / 256, 150 / 256, 197 / 256])

custom_cmap = LinearSegmentedColormap.from_list("custom", [start_color, end_color])


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    pred, var = torch.load(file_path, map_location="cpu")
    return pred.numpy(), var.numpy()


def generate_3d_plot(ax, model_data):
    x = np.linspace(0, 2, num_bins)
    y = np.arange(len(layer_counts))
    dx = 2 / num_bins
    dy = 0.25

    for i, data in enumerate(model_data):
        if data is not None:
            pred, var = data
            for j, (height, error) in enumerate(zip(pred, var)):
                color = custom_cmap(j / (num_bins - 1))  # Color based on eigenvalue
                ax.bar3d(
                    x[j],
                    y[i],
                    0,
                    dx,
                    dy,
                    height,
                    shade=True,
                    color=color,
                    alpha=0.8,
                    linewidth=2,
                )
                ax.plot(
                    [x[j], x[j]],
                    [y[i], y[i]],
                    [height, height + error],
                    color="black",
                    linewidth=1,
                )


fontsize = 48


def generate_3d_parameter_study(dataset):
    fig = plt.figure(figsize=(38, 8))
    fig.text(0.01, 0.5, "Accuracy", va="center", rotation="vertical", fontsize=54)

    for j, model in enumerate(models):
        ax = fig.add_subplot(1, 5, j + 1, projection="3d")

        model_data = []
        for layers in layer_counts:
            file_name = f"{dataset}/{model}_lr_0.001_epochs_500_hdim_{h_dim}_layers_{layers}_dout_0.0_binwidth_0.1.pt"
            file_path = os.path.join(path_root, file_name)
            data = load_data(file_path)
            model_data.append(data)

        generate_3d_plot(ax, model_data)

        ax.set_xlabel("Frequency", fontsize=fontsize, labelpad=35)
        ax.set_ylabel("Layers", fontsize=fontsize, labelpad=40)

        ax.zaxis.set_rotate_label(False)

        ax.set_title(model, fontsize=54, pad=10)
        ax.set_yticks(np.arange(len(layer_counts)) + 0.25)
        ax.set_yticklabels(layer_counts)
        ax.set_zlim(0, 1)

        ax.set_xticks([0, 1, 2])
        ax.set_xlim(2, 0)

        ax.set_xticklabels(
            ["0", "1", "2"], rotation=-15, ha="left", horizontalalignment="right"
        )

        ax.tick_params(axis="both", which="major", labelsize=fontsize, pad=10)
        ax.view_init(elev=20, azim=45)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.05, wspace=0.2)

    save_path = os.path.join(f"{dataset}_3d_parameter_study.pdf")
    plt.savefig(save_path, format="pdf", dpi=600)
    print(f"Saved to {save_path}")
    plt.close()


for dataset in datasets:
    generate_3d_parameter_study(dataset)

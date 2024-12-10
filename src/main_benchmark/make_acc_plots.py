import torch
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

path_root = "./results"
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["mathtext.default"] = "regular"
sb.set_palette("tab10")

start_color = np.array([230 / 256, 150 / 256, 59 / 256])
end_color = np.array([89 / 256, 150 / 256, 197 / 256])

all_gnns = [
    ["Transformer", "SAGE", "GCNII", "GATv2", "GAT", "SGC", "APPNP"],
    ["Cheb", "FA", "GCN", "GIN", "1-GNN", "GPS", "GatedGraph"],
]

datasets = ["Photo", "DBLP", "CS", "Cora_Full", "Physics", "Computers"]

for idx, gnns in enumerate(all_gnns):
    fig, axs = plt.subplots(len(datasets), len(gnns), figsize=(18, 12), sharey=True)

    for i, data_name in enumerate(datasets):
        for j, gnn in enumerate(gnns):
            if gnn == "GCNII":
                gnn_name = "GCN2"
            elif gnn == "1-GNN":
                gnn_name = "Graph"
            else:
                gnn_name = gnn

            file = f"{gnn_name}_lr_0.001_epochs_500_hdim_64_layers_2_dout_0.0_bindwidth_0.1.pt"

            pred, var = torch.load(
                path_root + "/" + data_name + "/" + file, map_location="cpu"
            )

            ax = axs[i, j]

            num_bins = 20
            colors = [
                start_color + (end_color - start_color) * (k / (num_bins - 1))
                for k in range(num_bins)
            ]
            x = np.linspace(0, 2, num_bins)
            width = 2 / (num_bins * 1.2)

            ax.bar(
                x,
                pred,
                width=width,
                color=colors,
                alpha=0.73,
                edgecolor="black",
                linewidth=2,
                capsize=2,
                yerr=var,
            )

            ax.set_facecolor("#EEF0F2")
            ax.grid(True, linestyle="--", color="gray", alpha=0.5)
            ax.grid(axis="y", linestyle="--", alpha=0.7)

            if i == 0:
                ax.set_title(gnn, fontsize=30)
            if j == 0:
                if data_name == "Cora":
                    ax.set_ylabel("Cora-Full", fontsize=30, rotation=90)
                else:
                    ax.set_ylabel(data_name, fontsize=30, rotation=90)

            ax.set_xticks(np.arange(0, 2.1, 0.5))
            ax.set_xticklabels(np.arange(0, 2.1, 0.5), fontsize=20, rotation=35)

            ax.set_yticks(np.round(np.arange(0, 1.1, 0.2), 1))
            ax.set_yticklabels(np.round(np.arange(0, 1.1, 0.2), 1), fontsize=20)

    plt.tight_layout()
    plt.savefig(f"combined_plot_{idx}.pdf", bbox_inches="tight", format="pdf", dpi=600)
    plt.show()

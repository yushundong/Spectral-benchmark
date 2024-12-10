import pdb
import matplotlib.pyplot as plt
import numpy as np

from argparse import ArgumentParser
from rank_by_benchmark import get_benchmark_ranking

def get_average_ranking(rankings): # rankings: list of list of models
    models = rankings[0]
    avg_rankings = {}

    for model in models:
        positions = [ranking.index(model) + 1 for ranking in rankings]
        avg_rankings[model] = np.mean(positions)
    
    return avg_rankings


def main(args):
    if args.recompute:
        benchmark_rankings = get_benchmark_ranking()
        data = {}
        models = set(benchmark_rankings['overall'][0])

        low_rankings = get_average_ranking(benchmark_rankings['low'])
        mid_rankings = get_average_ranking(benchmark_rankings['mid'])
        high_rankings = get_average_ranking(benchmark_rankings['high'])

        for model in models:
            data[model] = [
                low_rankings[model],
                mid_rankings[model],
                high_rankings[model]
            ]
    else: # use reported rankings
        data = {
            "SAGE": [2.67, 6.67, 3],
            "GCN2": [3.83, 4, 4.83],
            "GATv2": [4.17, 4, 6.83],
            "Graph": [4.83, 5.33, 5.17],
            "Transformer": [5, 9.17, 3.83],
            "Cheb": [5.17, 8.33, 6],
            "GCN": [7, 4, 4.83],
            "GPS": [7.83, 10.5, 6],
            "GAT": [8.17, 5.83, 9.5],
            "SGC": [8.83, 5.5, 6.67],
            "GIN": [9.5, 5.67, 10.83],
            "FA": [11.33, 9.5, 12.83],
            "APPNP": [12.67, 12.5, 12.67],
            "GatedGraph": [14, 14, 12],
        }

    # X-axis labels
    x_labels = ["Low", "Mid", "High"]
    x = np.arange(len(x_labels))

    # Create the figure and subplots
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 25
    # List of marker styles and colors
    markers = ["o", "s", "D", "^", "v", "p", "*", "<", ">", "P", "X", "h", "d", "H"]
    colors = plt.cm.tab20.colors
    grey_color = "#a6a6a6"

    # Create an empty list to collect handles and labels for the legend
    legend_handles = []
    legend_labels = []

    # Iterate over each subplot
    for j, (ax, focus_idx) in enumerate(zip(axs, range(3))):
        ax.set_facecolor("#eef0f2")  # Set background to light grey
        ax.grid(True, color="black", linestyle=":")  # Set grid lines

        # Plot each model's data
        for i, (model, rankings) in enumerate(data.items()):
            min_idx = np.argmin(
                rankings
            )  # Find index of the minimum value in the rankings

            if min_idx == focus_idx:
                (line,) = ax.plot(
                    x,
                    rankings,
                    label=model,
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    alpha=0.5,
                    markerfacecolor=colors[i % len(colors)],
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    markersize=30,
                    linewidth=5,
                )
                legend_handles.append(line)
                if model == "GCN2":
                    model = "GCNII"
                if model == "Graph":
                    model = "1-GNN"
                legend_labels.append(model)
            else:
                line = ax.scatter(
                    x,
                    rankings,
                    label=model,
                    marker=markers[i % len(markers)],
                    color=grey_color,
                    alpha=0.2,
                    s=900,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=25)
        ax.set_xlabel("Frequency Component Ranges", fontsize=25)
        ax.set_ylabel("Avg Ranking", fontsize=25)

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.21),
        ncol=5,
        markerscale=0.4,
        fontsize=25,
        handletextpad=0.5,
        columnspacing=4.5,
        labelspacing=0.2,
    )

    plt.tight_layout()
    plt.savefig("ranking_performance.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute the rankings, otherwise use the provided rankings",
    )
    args = parser.parse_args()
    main(args)


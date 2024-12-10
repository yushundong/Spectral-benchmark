import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

from argparse import ArgumentParser
from rank_by_benchmark import get_benchmark_ranking
from rank_by_downstream import get_downstream_ranking


def kt_distance(perm1, perm2):  # kt = kendall tau
    if len(perm1) != len(perm2):
        raise ValueError("Permutations must have the same length")
    n = len(perm1)
    inversions = 0
    for i in range(n):
        for j in range(i + 1, n):
            # Check if the relative order of elements i and j is different in perm1 and perm2
            if (perm1[i] - perm1[j]) * (perm2[i] - perm2[j]) < 0:
                inversions += 1
    return inversions


def get_kts(rankings, gt_rankings):
    kts_means = {}

    for test_dataset, gt_ranking in gt_rankings.items():
        kts = []
        for ranking in rankings:
            gt_ordinal = list(range(len(gt_ranking)))
            ranking_ordinal = [gt_ranking.index(gnn) for gnn in ranking]
            kt = kt_distance(gt_ordinal, ranking_ordinal)
            kts.append(kt)

        kts_means[test_dataset] = {
            "mean": np.mean(kts),
            "std": np.std(kts),
        }

    return kts_means


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 40
sb.set_palette("tab10")
start_color = np.array([230 / 256, 150 / 256, 59 / 256])
end_color = np.array([89 / 256, 150 / 256, 197 / 256])
green_color = np.array([92 / 256, 164 / 256, 105 / 256])

dataset_to_spectrum = {
    "Airports_Brazil": "overall",
    "Wisconsin": "overall",
    "Cornell": "overall",
    "Texas": "overall",
    "squirrel": "overall",
    "chameleon": "overall",
}


def main(args):
    if args.recompute:
        benchmark_rankings = get_benchmark_ranking()
        _, downstream_rankings = get_downstream_ranking()
        _, gt_rankings = get_downstream_ranking(is_test=True)

        downstream_kts = get_kts(list(downstream_rankings.values()), gt_rankings)
        mean_downstream_kts, downstream_kt_stds = [], []
        for test_dataset in dataset_to_spectrum:
            mean_downstream_kts.append(downstream_kts[test_dataset]["mean"])
            downstream_kt_stds.append(downstream_kts[test_dataset]["std"])

        overall_rankings = benchmark_rankings["overall"]
        low_rankings = benchmark_rankings["low"]
        mid_rankings = benchmark_rankings["mid"]
        high_rankings = benchmark_rankings["high"]

        overall_kts = get_kts(overall_rankings, gt_rankings)
        low_kts = get_kts(low_rankings, gt_rankings)
        mid_kts = get_kts(mid_rankings, gt_rankings)
        high_kts = get_kts(high_rankings, gt_rankings)

        kt_dict = {
            "overall": overall_kts,
            "low": low_kts,
            "mid": mid_kts,
            "high": high_kts,
        }

        mean_benchmark_kts, benchmark_kt_stds = [], []

        for test_dataset in dataset_to_spectrum:
            kts = kt_dict[dataset_to_spectrum[test_dataset]]
            mean = kts[test_dataset]["mean"]
            std = kts[test_dataset]["std"]
            mean_benchmark_kts.append(mean)
            benchmark_kt_stds.append(std)

        num_models = len(list(gt_rankings.values())[0])
        random_ranking = num_models * (num_models - 1) / 4

    else:  # reported numbers
        random_ranking = 45.5  # 14 models => 14*13/2 = 91, 91/2 = 45.5 due to symmetry
        mean_downstream_kts = [
            52.166666666666664,
            37.5,
            41.833333333333336,
            38.833333333333336,
            31.5,
            35.5,
        ]
        downstream_kt_stds = [
            2.6718699236468995,
            10.719919153924002,
            10.25372561018135,
            12.508885730640527,
            12.945398152754258,
            11.29527924990495,
        ]

        mean_benchmark_kts = [
            31.833333333333332,
            34.5,
            37.833333333333336,
            35.166666666666664,
            34.166666666666664,
            35.166666666666664,
        ]
        benchmark_kt_stds = [
            2.4094720491334933,
            3.8622100754188224,
            4.94694069321861,
            3.8042374035044424,
            13.132868012061273,
            9.753916592266355,
        ]

        airports_brazil_ranking = [
            "GatedGraph",
            "Transformer",
            "GPS",
            "GIN",
            "APPNP",
            "SAGE",
            "GCN",
            "FA",
            "GAT",
            "GCN2",
            "Graph",
            "SGC",
            "Cheb",
            "GATv2",
        ]
        texas_ranking = [
            "GPS",
            "GatedGraph",
            "SAGE",
            "Cheb",
            "Graph",
            "Transformer",
            "FA",
            "GCN2",
            "GATv2",
            "SGC",
            "GAT",
            "GCN",
            "GIN",
            "APPNP",
        ]
        wisconsin_ranking = [
            "Transformer",
            "GatedGraph",
            "SAGE",
            "Cheb",
            "Graph",
            "GPS",
            "GCN2",
            "FA",
            "GCN",
            "GATv2",
            "SGC",
            "APPNP",
            "GAT",
            "GIN",
        ]
        cornell_ranking = [
            "GatedGraph",
            "Graph",
            "Transformer",
            "GPS",
            "SAGE",
            "Cheb",
            "GCN2",
            "GCN",
            "GATv2",
            "FA",
            "GIN",
            "APPNP",
            "GAT",
            "SGC",
        ]
        squirrel_ranking = [
            "SAGE",
            "Cheb",
            "Transformer",
            "GPS",
            "GatedGraph",
            "GATv2",
            "FA",
            "GAT",
            "GCN2",
            "GCN",
            "APPNP",
            "SGC",
            "GIN",
            "Graph",
        ]
        chameleon_ranking = [
            "Transformer",
            "SAGE",
            "Cheb",
            "GPS",
            "GATv2",
            "FA",
            "APPNP",
            "GatedGraph",
            "GAT",
            "Graph",
            "GCN2",
            "SGC",
            "GIN",
            "GCN",
        ]

        gt_rankings = {
            "Airports_Brazil": airports_brazil_ranking,
            "Wisconsin": wisconsin_ranking,
            "Cornell": cornell_ranking,
            "Texas": texas_ranking,
            "squirrel": squirrel_ranking,
            "chameleon": chameleon_ranking,
        }

    datasets = [
        "Airports_Brazil",
        "Wisconsin",
        "Cornell",
        "Texas",
        "Squirrel",
        "Chameleon",
    ]

    datasets_brief = [
        "AB",
        "Wisc.",
        "Corn.",
        "Texas",
        "Squi.",
        "Cham.",
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    if args.recompute:
        llim = min(min(mean_downstream_kts), min(mean_benchmark_kts))
        ulim = max(max(mean_downstream_kts), max(mean_benchmark_kts)) + 2
        ax.set_ylim([llim, ulim])
    else:
        ax.set_ylim([25, 55])
    plt.xticks(np.arange(0, 2.1, 0.2))
    plt.gca().set_facecolor("#EEF0F2")
    plt.grid(True, linestyle="--", color="gray", alpha=0.5)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    width = 0.25
    x = np.arange(len(datasets))

    rects1 = ax.bar(
        x - width,
        [random_ranking] * len(datasets),
        width,
        label="Random Ranking τ",
        color=start_color,
        alpha=0.73,
        edgecolor="black",
        linewidth=4,
        capsize=2,
    )
    rects2 = ax.bar(
        x + width,
        mean_benchmark_kts,
        width,
        label="Benchmark Ranking τ",
        color=green_color,
        alpha=0.73,
        edgecolor="black",
        linewidth=4,
        capsize=2,
    )
    rects3 = ax.bar(
        x,
        mean_downstream_kts,
        width,
        label="Original Task Ranking τ",
        color=end_color,
        alpha=0.73,
        edgecolor="black",
        linewidth=4,
        capsize=2,
    )

    # Add error bars
    ax.errorbar(
        x + width,
        mean_benchmark_kts,
        yerr=np.sqrt(benchmark_kt_stds),
        fmt="none",
        capsize=5,
    )
    ax.errorbar(
        x,
        mean_downstream_kts,
        yerr=np.sqrt(downstream_kt_stds),
        fmt="none",
        capsize=5,
    )

    ax.tick_params(axis="both", which="major", labelsize=30)
    ax.tick_params(axis="both", which="minor", labelsize=30)
    ax.set_ylabel(r"Kendall's τ", fontsize=40)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets_brief, fontsize=40)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.45, 1.5),
        ncol=2,
        fontsize=30,
        columnspacing=0.001,
    )

    fig.tight_layout()
    plt.savefig("kt_distance.pdf", bbox_inches="tight", format="pdf", dpi=600)
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

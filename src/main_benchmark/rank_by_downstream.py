import os
import torch
import pandas as pd


def get_downstream_ranking(is_test=False):
    results = {}

    if is_test:
        directory = "./downstream_test_results"
    else:
        directory = "./downstream_results"

    for root, dirs, files in os.walk(directory):
        for file in files:
            if not file.endswith(".pt"):
                continue

            data = torch.load(os.path.join(root, file))
            by_under = file.split("_")

            if "Airports" in by_under:
                model = by_under[0]
                dataset = "Airports_" + by_under[2]
            else:
                model, dataset = by_under[0], by_under[1]

            test_accs = torch.tensor([x["test_acc"] for x in data])
            avg_test_acc = test_accs.mean().item()
            std_dev = test_accs.std().item()

            if model not in results:
                results[model] = {}

            results[model][dataset] = (avg_test_acc, std_dev)

    ranking = {}

    for dataset in set(
        dataset for model_data in results.values() for dataset in model_data.keys()
    ):
        dataset_results = []

        for model in results:
            if dataset in results[model]:
                avg_test_acc, std_dev = results[model][dataset]
                dataset_results.append((model, avg_test_acc, std_dev))

        dataset_results.sort(key=lambda x: x[1], reverse=True)
        ranking[dataset] = dataset_results

    for dataset, ranked_models in ranking.items():
        print(f"Ranking for dataset: {dataset}")
        print(f"{'Rank':<5} {'Model':<15} {'Avg Test Acc':<15} {'Std Dev':<10}")
        for rank, (model, avg_acc, std_dev) in enumerate(ranked_models, start=1):
            print(f"{rank:<5} {model:<15} {avg_acc:<15.4f} {std_dev:<10.4f}")
        print("\n")

    ranking_list = []
    for dataset, ranked_models in ranking.items():
        for rank, (model, avg_acc, std_dev) in enumerate(ranked_models, start=1):
            ranking_list.append(
                {
                    "Dataset": dataset,
                    "Rank": rank,
                    "Model": model,
                    "Avg Test Acc": avg_acc,
                    "Std Dev": std_dev,
                }
            )

    # compute average rank per model per dataset
    model_ranks = {}
    for dataset, ranked_models in ranking.items():
        for rank, (model, avg_acc, std_dev) in enumerate(ranked_models, start=1):
            if model not in model_ranks:
                model_ranks[model] = []
            model_ranks[model].append(rank)

    model_avg_ranks = {
        model: sum(ranks) / len(ranks) for model, ranks in model_ranks.items()
    }
    model_avg_ranks = sorted(model_avg_ranks.items(), key=lambda x: x[1])

    df_ranking = pd.DataFrame(ranking_list)

    if is_test:
        fname = "downstream_test_ranking.csv"
    else:
        fname = "downstream_ranking.csv"

    df_ranking.to_csv(fname, index=False)
    overall_ranking = [x[0] for x in model_avg_ranks]

    ranking_by_dataset = {}
    for dataset, ranked_models in ranking.items():
        ranking_by_dataset[dataset] = [x[0] for x in ranked_models]

    return overall_ranking, ranking_by_dataset

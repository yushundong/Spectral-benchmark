import pdb
import torch
import pandas as pd
import numpy as np


def calculate_stats(gnn: str, rankings):
    positions = [ranking.index(gnn) + 1 for ranking in rankings]
    avg = np.mean(positions)

    return avg


def analyze_rankings(spectrum_rankings):
    gnns = set(spectrum_rankings[0])  # Assuming all lists contain the same GNNs
    results = {}

    for gnn in gnns:
        avg = calculate_stats(gnn, spectrum_rankings)
        results[gnn] = avg

    results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1])}
    return results


def analyze_all_spectrums(overall, low, mid, high):
    spectrums = {"overall": overall, "low": low, "mid": mid, "high": high}

    all_results = {}

    for spectrum_name, rankings in spectrums.items():
        all_results[spectrum_name] = analyze_rankings(rankings)

    return all_results


def parse_sheets(sheets):
    data = {}
    for sheet_name, df in sheets.items():
        frequency = sheet_name
        data[frequency] = {}
        for dataset, row in df.iterrows():
            data[frequency][dataset] = row.to_dict()
    return data


def rank_models(data):
    rankings = {}
    for frequency, freq_data in data.items():
        rankings[frequency] = {}
        for dataset, scores in freq_data.items():
            sorted_models = sorted(
                scores.items(), key=lambda x: float(x[1].split("±")[0]), reverse=True
            )
            rankings[frequency][dataset] = [model for model, _ in sorted_models]
    return rankings


def get_benchmark_ranking():
    models = [
        "GatedGraph",
        "GCN",
        "Cheb",
        "GAT",
        "GCN2",
        "FA",
        "GATv2",
        "GIN",
        "SGC",
        "Graph",
        "Transformer",
        "SAGE",
        "GPS",
        "APPNP",
    ]
    datas = ["Cora_Full", "Computers", "CS", "DBLP", "Photo", "Physics"]

    out_table = [
        [["" for _ in range(4)] for _ in range(len(models))] for _ in range(len(datas))
    ]
    data_root = "./results"

    for i in range(len(datas)):
        for j in range(len(models)):
            file_path = (
                f"{data_root}/{datas[i]}/"
                + f"{models[j]}_lr_0.001_epochs_500_hdim_64_layers_2_dout_0.0_bindwidth_0.1.pt"
            )

            try:
                model_data = torch.load(file_path)[0]
            except:
                continue

            l_b = len(model_data)
            mean_total = (torch.mean(model_data) * 100).item()
            var_total = (torch.var(model_data) * 100).item()
            mean_first = (torch.mean(model_data[: int(1 / 3 * l_b)]) * 100).item()
            var_first = (torch.var(model_data[: int(1 / 3 * l_b)]) * 100).item()

            mean_second = (
                torch.mean(model_data[int(1 / 3 * l_b) : int(2 / 3 * l_b)]) * 100
            ).item()
            var_second = (
                torch.var(model_data[int(1 / 3 * l_b) : int(2 / 3 * l_b)]) * 100
            ).item()

            mean_third = (torch.mean(model_data[int(2 / 3 * l_b) : l_b]) * 100).item()
            var_third = (torch.var(model_data[int(2 / 3 * l_b) : l_b]) * 100).item()

            out_table[i][j][0] = f"{mean_total:.2f} ± {var_total:.2f}"
            out_table[i][j][1] = f"{mean_first:.2f} ± {var_first:.2f}"
            out_table[i][j][2] = f"{mean_second:.2f} ± {var_second:.2f}"
            out_table[i][j][3] = f"{mean_third:.2f} ± {var_third:.2f}"

    sheets = {}

    for k in range(4):
        data_slice = [
            [out_table[i][j][k] for j in range(len(models))] for i in range(len(datas))
        ]
        df = pd.DataFrame(data_slice, index=datas, columns=models)
        sheets[f"Frequency_{k+1}"] = df

    file_path = "benchmark_ranking.csv"
    pd.concat(sheets).to_csv(file_path)
    print(f"Benchmark ranking saved to {file_path}")

    ranked_by_freq = rank_models(parse_sheets(sheets))

    overall_freq_rankings = list(ranked_by_freq["Frequency_1"].values())
    low_freq_rankings = list(ranked_by_freq["Frequency_2"].values())
    mid_freq_rankings = list(ranked_by_freq["Frequency_3"].values())
    high_freq_rankings = list(ranked_by_freq["Frequency_4"].values())

    return {
        "overall": overall_freq_rankings,
        "low": low_freq_rankings,
        "mid": mid_freq_rankings,
        "high": high_freq_rankings,
    }

if __name__ == "__main__":
    get_benchmark_ranking()
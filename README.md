# Graph Neural Networks Are More Than Filters

This repository holds the code for the research work “Graph Neural Networks Are More Than Filters: Revisiting and Benchmarking From a Spectral Perspective” published in ICLR 2025.

## Citation
```
@inproceedings{dong2025graph,
  title={Graph Neural Networks Are More Than Filters: Revisiting and Benchmarking from A Spectral Perspective},
  author={Dong, Yushun and Soga, Patrick and He, Yinhan and Wang, Song and Li, Jundong},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```

## Usage

To run the code, first set up all the dependencies by running the following command:

```bash
python3 -m venv ./virtualenvs/spectral_benchmark
source ./virtualenvs/spectral_benchmark/bin/activate
pip3 install -r requirements.txt
```
Now run `cd src` and follow the instructions below.

## Exploratory Study Details

### Figures 1 & 7 &ndash; 10
These plots belong to the exploratory study, showcasing the capability of GNNs to shift energy distributions between frequencies. To generate the exploratory study plots, run `run_exploratory.sh`. All generated plots and model outputs will be saved in the `exploratory_study/results` directory.

## Main benchmark

To run the main benchmark, run `run_main.sh`. All the model results will be saved the in `main_benchmark/results` directory. These files will be used to generate the majority of the figures and tables in the paper.

To use the numbers from our work, you can download the results from [this anonymous Google Drive link](https://drive.google.com/file/d/1S3oMSa9nanK0PkAdKZ1UHAXpCLsumKhF/view?usp=sharing) and extract the contents into the `main_benchmark/results` directory with `unzip results.zip -d main_benchmark`.

### Figures 2 & 12 &ndash; 13

These figures showcase GNN performance across the whole spectral domain for various datasets. To reproduce these plots, run `make_acc_plots.py`.

### Tables 1 & 3 &ndash; 5

These tables show the ranking of GNNs on the main benchmark task per spectrum region. To compute these tables, run `rank_by_benchmark.py` which will save the results in `main_benchmark/benchmark_ranking.csv`.

### Figure 3

This figure is a performance comparison in the average ranking of 14 GNNs on six real-world datasets according to low, middle, and high frequencies. To produce this figure, run `make_ranking_plot.py --recompute`, and omit the `--recompute` flag to generate the figure using our numbers. The resulting figure will be saved in `main_benchmark/ranking_performance.pdf`.

### Figure 4

This figure compares the ranking produced by the low-spectrum performance portion of our benchmark with a ranking based on node classification accuracy on those same datasets (dubbed the *downstream* ranking in this README). Specifically, we first compute the ranking of various GNNs based on their node classification accuracies on a set of held-out datasets. Then, we compare the Kendall-$\tau$ distance between our benchmark ranking and this held-out ranking with the Kendall-$\tau$ distance between the downstream ranking and the held-out ranking. For context, we also include the Kendall-$\tau$ distance of a random ranking and the held-out ranking.

Producing this figure assumes you have already run the main benchmark.

1. First, produce the downstream task ranking by running `run_downstream.sh`. All the model results will be stored in `main_benchmark/downstream_results`.
2. Then, run `compute_rankings.py --recompute` to produce the downstream ranking stored at `main_benchmark/downstream_ranking.csv` and then generate the figure at `main_benchmark/kt_distance.pdf`. To generate the figure from the paper using our numbers, omit the `--recompute` flag.

### Figures 5 & 14

These figures are a parameter study on the main benchmark results w.r.t layer depth and hidden dimension size. First, run `run_parameter_study.sh` in order to generate the results. Then, run `main_benchmark/parameter_study.py` to save the figures in `main_benchmark/{dataset}_3d_parameter_study.pdf`.

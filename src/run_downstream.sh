export PYTHONPATH=$(pwd):$PYTHONPATH

cd main_benchmark;

models=( GCN GCN2 GAT GATv2 GIN Transformer SGC FA SAGE Cheb Graph GatedGraph GPS APPNP )
datasets=( Photo Computers Products Physics CS Cora_Full DBLP )

hdim=64
drop=0.0
layer_num=2

for i in "${!models[@]}"; do
    for d in "${!datasets[@]}"; do
        model=${models[$i]};
        dataset=${datasets[$d]};

        log_dir=downstream_logs/${dataset}/${model};
        mkdir -p $log_dir;
        echo "Running $model on $dataset, saving to $log_dir";

        python3 downstream.py --dataset $dataset \
            --lr 0.0001 \
            --epochs 500 \
            --num_exp_times 3 \
            --dropout $drop \
            --device 0 \
            --h_dim $hdim \
            --gnn $model \
            --layers $layer_num \
            --results_dir downstream_results \
        1>$log_dir/$model-$dataset-$hdim-$drop-$layer_num.log 2>$log_dir/$model-$dataset-$hdim-$drop-$layer_num.err
    done
done

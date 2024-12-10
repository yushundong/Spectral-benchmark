export PYTHONPATH=$(pwd):$PYTHONPATH

models=( GCN GCN2 GAT GATv2 GIN Transformer SGC FA SAGE Cheb Graph GatedGraph GPS APPNP )
datasets=( CS Computers Physics Photo Cora_Full DBLP )

mask_ratios_train=( 0.9 )
mask_ratios_val=( 0.05 )
mask_ratios_test=( 0.05 )

frequency_cutoff_low=( 0.33 )
frequency_cutoff_med=( 0.33 )
frequency_cutoff_high=( 0.33 )

bins=( 20 )

cd exploratory_study;
for bin in "${bins[@]}";
do
    for dataset in "${datasets[@]}";
    do
        for model in "${models[@]}";
        do
            for i in {0..2};
            do
                train_ratio=0.9
                val_ratio=0.05
                test_ratio=0.05
                freq_low=0.33
                freq_med=0.33
                freq_high=0.33

                freq_str="freq-${freq_low}-${freq_med}-${freq_high}"
                ratio_str="ratio-${train_ratio}-${val_ratio}-${test_ratio}"

                tag_name="${dataset}_${model}_${ratio_str}_${freq_str}_bins-${bin}"
                python3 main.py --gnn_model $model \
                 --dataset $dataset \
                 --seeds 42 43 44 \
                 --h_dim 64 \
                 --layers 3 \
                 --frequency_cutoffs $freq_low $freq_med $freq_high \
                 --feat_dim -1 \
                 --train_val_test_ratios $train_ratio $val_ratio $test_ratio \
                 --tag $tag_name \
                 --bin_width $bin \
                 --dropout 0.0 \
                 --epochs 100 \
                 --lr 0.0002 \
                 --device 0
            done
        done
    done
done

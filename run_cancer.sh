for model_name in sparse_feature_linear sparse_feature_net
do
    for alpha in 1e-3 3e-3 1e-2 3e-2 1e-1 3e-1
    do
        for optname in Adam SGD
        do
            python3 cancer_sparse_classification.py \
                --model_name=$model_name --alpha=$alpha --optname=$optname --log_dir=output/cancer_${model_name}_alpha=${alpha}_${optname}_lr1e-1 \
                --lr=1e-1 --device="cuda:0" &
            python3 cancer_sparse_classification.py \
                --model_name=$model_name --alpha=$alpha --optname=$optname --log_dir=output/cancer_${model_name}_alpha=${alpha}_${optname}_lr1e-2 \
                --lr=1e-2 --device="cuda:1" &
            python3 cancer_sparse_classification.py \
                --model_name=$model_name --alpha=$alpha --optname=$optname --log_dir=output/cancer_${model_name}_alpha=${alpha}_${optname}_lr1e-3 \
                --lr=1e-3 --device="cuda:2" &
            python3 cancer_sparse_classification.py \
                --model_name=$model_name --alpha=$alpha --optname=$optname --log_dir=output/cancer_${model_name}_alpha=${alpha}_${optname}_lr1e-4 \
                --lr=1e-4 --device="cuda:3" &
            wait
        done
    done
done
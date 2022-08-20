for predictor_dim in 1000 10000 100000
do
    for respond_dim in 1 1000
    do
        python3 high_dim_regression.py \
            --device=cuda:1 \
            --optname=SGD \
            --lr=1e-2 \
            --predictor_dim=$predictor_dim \
            --respond_dim=$respond_dim \
            --epoch=10000 \
            --num_alpha=5 \
            --output_folder=output/HighDimLinearRegression/${predictor_dim}_${respond_dim}_${reg}
    done
done



for predictor_dim in 1000000 10000000
do
    for respond_dim in 1 1000
    do
        python3 high_dim_regression.py \
            --device=cuda:1 \
            --optname=SGD \
            --lr=1e-3 \
            --predictor_dim=$predictor_dim \
            --respond_dim=$respond_dim \
            --epoch=10000 \
            --num_alpha=5 \
            --output_folder=output/HighDimLinearRegression/${predictor_dim}_${respond_dim}_${reg}
    done
done
for alpha in 1 10 20 40 80 160 320
do
    python3 classic_logistic_regression.py --alpha=$alpha --dataset=20news --logging_path output/SparseLogisticRegression/20news_lr.log &
    python3 classic_logistic_regression.py --alpha=$alpha --dataset=mnist --logging_path output/SparseLogisticRegression/mnist_lr.log
    wait
done


# for alpha in 1e-6 1e-5 1e-4 1e-3 1e-2 1 10 100 1000
# do
#     python3 neural_logistic_regression.py --alpha=$alpha --dataset=20news --logging_path output/SparseLogisticRegression/20news_sparse_weight.log
# done
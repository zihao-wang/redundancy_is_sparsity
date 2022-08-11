python3 sparse_linear_regression.py --device=cuda:0 --predictor_dim=100 --output_folder=output/100-1  &
python3 sparse_linear_regression.py --device=cuda:0 --respond_dim=100 --predictor_dim=100 --output_folder=output/100-100 &

wait

python3 plot.py --data_folder output/100-1
python3 plot.py --data_folder output/100-100
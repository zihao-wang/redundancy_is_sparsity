# respond_dim=1

# for optname in Adam SGD
# do
#     for lr in 5e-1 5e-2 5e-3
#     # for lr in 1 1e-1 1e-2
#     do
#         python3 high_dim_regression.py \
#             --device=cuda:0 \
#             --optname=$optname \
#             --lr=$lr \
#             --predictor_dim=100 \
#             --respond_dim=$respond_dim \
#             --num_alpha=5 &

#         python3 high_dim_regression.py \
#             --device=cuda:1 \
#             --predictor_dim=1000 \
#             --optname=$optname \
#             --lr=$lr \
#             --respond_dim=$respond_dim \
#             --num_alpha=5 &

#         python3 high_dim_regression.py \
#             --device=cuda:2 \
#             --optname=$optname \
#             --lr=$lr \
#             --predictor_dim=10000 \
#             --respond_dim=$respond_dim \
#             --num_alpha=5 &

#         python3 high_dim_regression.py \
#             --device=cuda:3 \
#             --predictor_dim=100000 \
#             --optname=$optname \
#             --lr=$lr \
#             --respond_dim=$respond_dim \
#             --num_alpha=5 &
#         wait
#     done
# done

respond_dim=100

# for optname in Adam SGD
# do
#     # for lr in 1 1e-1 1e-2
#     for lr in 5e-1 5e-2 5e-3
#     do
#         python3 high_dim_regression.py \
#             --device=cuda:0 \
#             --optname=$optname \
#             --lr=$lr \
#             --predictor_dim=100 \
#             --respond_dim=$respond_dim \
#             --num_alpha=5 &

#         python3 high_dim_regression.py \
#             --device=cuda:1 \
#             --predictor_dim=1000 \
#             --optname=$optname \
#             --lr=$lr \
#             --respond_dim=$respond_dim \
#             --num_alpha=5 &

#         python3 high_dim_regression.py \
#             --device=cuda:2 \
#             --optname=$optname \
#             --lr=$lr \
#             --predictor_dim=10000 \
#             --respond_dim=$respond_dim \
#             --num_alpha=5 &

#         python3 high_dim_regression.py \
#             --device=cuda:3 \
#             --predictor_dim=100000 \
#             --optname=$optname \
#             --lr=$lr \
#             --respond_dim=$respond_dim \
#             --num_alpha=5 &
#         wait
#     done
# done

for optname in Adam SGD
do
    # for lr in 1 1e-1 1e-2
    for lr in 5e-1 5e-2 5e-3
    do
        python3 high_dim_regression.py \
            --device=cuda:3 \
            --predictor_dim=100000 \
            --optname=$optname \
            --lr=$lr \
            --respond_dim=$respond_dim \
            --num_alpha=5 &
    done
done
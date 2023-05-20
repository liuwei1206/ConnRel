# For Ji split, change the dataset to pdtb2 or pdtb3
# 2
<<"COMMENT"
python3 train_pipeline.py --do_train \
                          --dataset="test" \
                          --label_file="labels_level_1.txt"
COMMENT
# 8

for seed in 106524 106464 106537 219539 430683
do
    python3 train_pipeline.py --do_train \
                              --dataset="pdtb2" \
                              --label_file="labels_level_1.txt" \
                              --seed=${seed}
done

sleep 10s

for seed in 106524 106464 106537 219539 430683
do
    python3 train_pipeline.py --do_train \
                              --dataset="pdtb2" \
                              --label_file="labels_level_2.txt" \
                              --seed=${seed}
done

sleep 10s

# For xval, change the dataset to pdtb2 or pdtb3
# 11
# <<"COMMENT"
for idx in 1 2 3 4 5 6 7 8 9 10 11 12
do
    python3 train_pipeline.py --do_train \
                              --dataset="pdtb2" \
                              --fold_id=${idx} \
                              --label_file="labels_level_1.txt"
done
# COMMENT
# 21

sleep 10s

for idx in 1 2 3 4 5 6 7 8 9 10 11 12
do
    python3 train_pipeline.py --do_train \
                              --dataset="pdtb2" \
                              --fold_id=${idx} \
                              --label_file="labels_level_2.txt"
done


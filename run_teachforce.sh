# For Ji split, change the dataset to pdtb2 or pdtb3
# 2
<<"COMMENT"
python3 train_roberta.py --do_train \
                         --dataset="test" \
                         --label_file="labels_level_1.txt" \
                         --use_conn \
                         --teacher_forcing
COMMENT
# 10

# 12
# <<"COMMENT"
for seed in 106524 106464 106537 219539 430683
do
    python3 train_roberta.py --do_train \
                             --dataset="pdtb2" \
                             --label_file="labels_level_1.txt" \
                             --seed=${seed} \
                             --use_conn \
                             --teacher_forcing
done
# COMMENT
# 24

sleep 10s

# 28
# <<"COMMENT"
for seed in 106524 106464 106537 219539 430683
do
    python3 train_roberta.py --do_train \
                             --dataset="pdtb2" \
                             --label_file="labels_level_2.txt" \
                             --seed=${seed} \
                             --use_conn \
                             --teacher_forcing
done
# COMMENT
# 40

sleep 10s

# For xval, change the dataset to pdtb2 or pdtb3
# 15
# <<"COMMENT"
for idx in 1 2 3 4 5 6 7 8 9 10 11 12
do
    python3 train_roberta.py --do_train \
                             --dataset="pdtb2" \
                             --fold_id=${idx} \
                             --label_file="labels_level_1.txt" \
                             --use_conn \
                             --teacher_forcing
done
# COMMENT
# 57

sleep 10s

# 61
# <<"COMMENT"
for idx in 1 2 3 4 5 6 7 8 9 10 11 12
do
    python3 train_roberta.py --do_train \
                             --dataset="pdtb2" \
                             --fold_id=${idx} \
                             --label_file="labels_level_2.txt" \
                             --use_conn \
                             --teacher_forcing
done
# COMMENT
# 74


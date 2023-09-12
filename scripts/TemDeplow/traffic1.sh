if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=TemDephigh

root_path_name=./data/Traffic/
data_path_name=Traffic.csv
model_id_name=Traffic
data_name=Traffic

random_seed=2021
# seq_len=672

for pred_len in 96 192 336 720
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$s   eq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --target OT\
      --seq_len 672 \
      --pred_len $pred_len \
      --enc_in 862 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --seq_kernel 336\
      --kernel_size 1\
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 50\
      --patience 20 \
      --gpu '0'\
      --devices '0'\
      --itr 1 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'xiaorong.log 
done
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
model_name=TemDeplow

root_path_name=./data/ETT/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

random_seed=2021

for model_name in Conv2d
do
    for pred_len in 96 192 192 336 
    do
        python -u run_longExp.py \
          --random_seed $random_seed \
          --is_training 1 \
          --root_path $root_path_name \
          --data_path $data_path_name \
          --model_id $model_id_name_$seq_len'_'$pred_len \
          --model $model_name \
          --data $data_name \
          --features M \
          --seq_len 420 \
          --pred_len $pred_len \
          --enc_in 7 \
          --seq_kernel 336\
          --kernel_size 336\
          --e_layers 3 \
          --n_heads 4 \
          --d_model 16 \
          --d_ff 128 \
          --dropout 0.03\
          --fc_dropout 0.03\
          --head_dropout 0\
          --patch_len 16\
          --stride 8\
          --des 'Exp' \
          --train_epochs 200\
          --patience 60 \
          --gpu '0'\
          --devices '0'\
          --itr 5 --batch_size 16 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'go.log 
    done
done
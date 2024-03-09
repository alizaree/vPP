CUDA_VISIBLE_DEVICES=0 python main.py \
   --optimizer 'adam' \
   --lr 0.005 \
   --dropout 0.2 \
   --batch_size 256 \
   --epochs 500 \
   --max_traj_len 3 \
   --M 2 \
   --aug_range 0 \
   --attn_heads 32 \
   --num_layers 2 \
   --step_size 40 \
   --lr_decay 0.65 \
   --model_name 'coin' \
   --dataset 'coin' \
   --num_action 778 \
   --num_tasks 180 \
   --img_input_dim 512 \
   --text_input_dim 768 \
   --embed_dim 128 \
   --root_dir 'dataset/coin' \
   --train_json 'dataset/coin/coin_train.json' \
   --valid_json 'dataset/coin/coin_valid.json' \
   --features_dir 'data/coin_features/full_npy' \
   --saved_path 'checkpoints'

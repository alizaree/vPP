CUDA_VISIBLE_DEVICES=0 python main_script_gc.py \
    --optimizer 'adam' \
    --lr 0.004 \
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
    --model_name 'crosstask' \
    --dataset 'crosstask' \
    --num_action 133 \
    --num_tasks 18 \
    --img_input_dim 512 \
    --text_input_dim 768 \
    --embed_dim 128 \
    --root_dir 'dataset/crosstask/crosstask_release' \
    --train_json 'dataset/crosstask/cross_task_data_False.json' \
    --valid_json 'dataset/crosstask/cross_task_data_True.json' \
    --features_dir '/home/yulei/data/crosstask/processed_data/' \
    --split 'base' \
    --saved_path 'checkpoints' \
    --return_frames \
    --vid_dir "/home/yulei/data/crosstask/crosstask_videos/videos/" \
    --img_dir "/dvmm-filer3a/users/ali/Data/CrossTask/crosstask_frame_states"

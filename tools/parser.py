import argparse

## add argument
def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', 
                        default='model', type=str, 
                        help='model name')
    parser.add_argument("--weights_path",
                        type=str, default="./models/GenHowTo/weights/GenHowTo-STATES-96h-v1")
    parser.add_argument("--device",
                        type=str, default="cuda")
    parser.add_argument("--num_inference_steps",
                        type=int, default=50)
    parser.add_argument("--num_steps_to_skip",
                        type=int, default=None)
    parser.add_argument("--guidance_scale",
                        type=float, default=9.0)
    parser.add_argument('--max_traj_len', 
                        default=3, type=int, metavar='MAXTRAJ',
                        help='max length (default: 54)')
    parser.add_argument('--d_model', 
                        default=1024, type=int, metavar='MAXTRAJ',
                        help='dim of the model')
    parser.add_argument('--input_dim', 
                        default=1024, type=int, metavar='MAXTRAJ',
                        help='dim of the model')
    parser.add_argument('--vis_input_dim', 
                        default=64, type=int, metavar='MAXTRAJ',
                        help='dim of the model')
    parser.add_argument("--dropout_rate",
                        type=float, default=0.2)
    parser.add_argument('--n_layer', 
                        default=4, type=int, metavar='MAXTRAJ',
                        help='dim of the model')
    parser.add_argument('--n_head', 
                        default=4, type=int, metavar='MAXTRAJ',
                        help='dim of the model')
    parser.add_argument('--dataset', 
                        default='crosstask_howto100m', type=str, 
                        help='features')
    parser.add_argument('--num_action',
                        default=133, type=int,
                        help='number of action classes (crosstask: 133, coin: 778)')
    parser.add_argument('--num_tasks',
                        default=18, type=int,
                        help='number of tasks (crosstask: 18, coin: 778)')
    
    parser.add_argument('--epochs', 
                        default=500, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size', '-b', 
                        default=72, type=int,
                        metavar='N', help='mini-batch size (default: 72)')
    parser.add_argument('--dropout',
                        default=0.1, type=float,
                        help='dropout rate')
    parser.add_argument('--optimizer', 
                        default='adam', type=str, 
                        help='optimizer (default: sgd)')
    parser.add_argument('--lr', '--learning-rate', 
                        default=0.01, type=float, metavar='LR', 
                        help='initial learning rate')
    parser.add_argument('--step_size', 
                        default=20, type=int, metavar='LRSteps', 
                        help='epochs to decay learning rate')
    parser.add_argument('--lr_decay',
                        default=0.65, type=float,
                        help='learning weight decay')
    parser.add_argument('--weight_decay', '--wd', 
                        default=0.0001, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--scheduler_p',
                        default=5, type=int,
                        help='schedular paitiance')
    parser.add_argument('--scheduler_f',
                        default=0.3, type=float,
                        help='schedular factor decay')
    parser.add_argument('--M',
                        default=1, type=int,
                        metavar='W', help='augmentation factor (default: 1)')
    parser.add_argument('--aug_range',
                        default=0, type=int,
                        metavar='W', help='augmentation range (default: 0)')
    parser.add_argument('--no_state_task', 
                        action='store_true',
                        help='inject task information into state decoder')

    parser.add_argument('--root_dir', 
                        default='/home/yulei/data/crosstask/crosstask_release', type=str, 
                        help='root dir')
    parser.add_argument('--train_json', 
                        default='/home/wenliang/procedure_planning/cross_task_data_False.json', type=str, 
                        help='train json file')
    parser.add_argument('--valid_json', 
                        default='/home/wenliang/procedure_planning/cross_task_data_True.json', type=str, 
                        help='valid json file')
    parser.add_argument('--features_dir', 
                        default='/home/yulei/data/crosstask/crosstask_features_clip_336px', type=str, 
                        help='features dir')
    parser.add_argument('--vid_dir', 
                        default='/home/yulei/data/crosstask/crosstask_videos/videos/', type=str, 
                        help='video dir')
    parser.add_argument('--img_dir', 
                        default='/dvmm-filer3a/users/ali/Data/CrossTask/crosstask_frame_states', type=str, 
                        help='state image dir')
    parser.add_argument('--embedding_dir', 
                        default='/dvmm-filer3a/users/ali/Data/CrossTask/genhowto_embeds/', type=str, 
                        help='genhowto embedding dir')
    parser.add_argument('--return_frames', 
                        action='store_true',
                        help='return frames of actions?')
    parser.add_argument('--save_image_states', 
                        action='store_true',
                        help='save_image_states mode')
    parser.add_argument('--save_embeddings', 
                        action='store_true',
                        help='save_embeddings mode')
    parser.add_argument('--eval', 
                        action='store_true',
                        help='evaluation mode')
    parser.add_argument('--saved_path', 
                        default='./logs/', type=str, 
                        help='descriptions dir')

    parser.add_argument('--last_epoch',
                        default=-1, type=int,
                        help='last epoch for adjusting learning rate')

    parser.add_argument('--split', 
                        default='base', type=str, 
                        help='split (base, p3iv)')

    parser.add_argument('--seed', 
                        default=3407, type=int, metavar='M',
                        help='queue size')

    parser.add_argument('--uncertain', 
                        action='store_true',
                        help='probabilistic model')
    parser.add_argument('--num_sample',
                         default=1500, type=int,
                         help='number of samples of noise-vectors')
    parser.add_argument('--checkpoint_path', 
                        default='/dvmm-filer3a/users/ali/Data/CheckPoints/vPP/', type=str, 
                        help='model checkpoint path')
    parser.add_argument('--checkpoint_name', 
                        default='None', type=str, 
                        help='model checkpoint path')

    return parser.parse_args()
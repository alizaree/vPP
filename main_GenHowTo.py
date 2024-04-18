import torch
import os
import time
import numpy as np

from utils import *
from metrics import *
from torch.utils.data import DataLoader
from models.procedure_model import ProcedureModel
from models.utils import AverageMeter
import wandb
from tools.parser import create_parser
from PIL import Image

from models.GenHowTo.genhowto_utils import load_genhowto_model, DDIMSkipScheduler



def run_genhowto(args):
    logger_path = "logs/{}_{}_len{}".format(
                    time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), 
                    args.model_name, 
                    args.max_traj_len)
    if args.last_epoch > -1:
        logger_path += "_last{}".format(args.last_epoch)
    os.makedirs(logger_path)
    log_file_path = os.path.join(logger_path, "log.txt")
    logger = get_logger(log_file_path)
    logger.info("{}".format(log_file_path))
    logger.info("{}".format(args))

    validate_interval = 1
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == 'crosstask':
        logger.info("Loading prompt features...")
        file_path = './data/descriptors_crosstask.json'
        # Load the JSON file
        with open(file_path, 'r') as f:
            state_prompts = json.load(f)
        #state_prompt_features = np.load(f'./data/state_description_features/crosstask_state_prompt_features.npy')

        ## parse raw data
        task_info_path = os.path.join(args.root_dir, "tasks_primary.txt")
        task_info = parse_task_info(task_info_path)
        with open("data/crosstask_idices.json", "r") as f:
            idices_mapping = json.load(f)
        anot_dir = os.path.join(args.root_dir, "annotations")
        anot_info = parse_annotation(anot_dir, task_info, idices_mapping)

        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(anot_info, args.img_dir, state_prompts, 
                                        args.train_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "train", M=args.M,
                                        vid_dir=args.vid_dir,
                                        save_image_states=args.save_image_states)
        
        logger.info("Loading valid data...")
        valid_dataset = ProcedureDataset(anot_info, args.img_dir, state_prompts, 
                                        args.valid_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "valid", M=args.M,
                                        vid_dir=args.vid_dir,
                                        save_image_states=args.save_image_states)
        transition_matrix = train_dataset.transition_matrix
        
    
    elif args.dataset == "coin":
        raise NotImplementedError
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/coin_state_prompt_features.npy')
    
        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "train", M=args.M)
        
        logger.info("Loading valid data...")
        valid_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.valid_json, args.max_traj_len, aug_range=args.aug_range, 
                                        mode = "valid", M=args.M)
        transition_matrix = train_dataset.transition_matrix

    elif args.dataset == "niv":
        raise NotImplementedError
        logger.info("Loading prompt features...")
        state_prompt_features = np.load(f'./data/state_description_features/niv_state_prompt_features.npy')

        logger.info("Loading training data...")
        train_dataset = ProcedureDataset(args.features_dir, state_prompt_features, 
                                        args.train_json, args.max_traj_len, num_action = 48,
                                        aug_range=args.aug_range, mode = "train", M=args.M)
        
        logger.info("Loading valid data...")
        valid_dataset = ProcedureDataset(args.features_dir, state_prompt_features,
                                        args.valid_json, args.max_traj_len, num_action = 48,
                                        aug_range=args.aug_range, mode = "valid", M=args.M)
        transition_matrix = train_dataset.transition_matrix

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    logger.info("Training set volumn: {} Testing set volumn: {}".format(len(train_dataset), len(valid_dataset)))
    # Initialize wandb
    #wandb_config=vars(args)
    #wandb.init(project='vPP', config=wandb_config)
    
    pipe = load_genhowto_model(args.weights_path, device=args.device)
    logger.info("the model is loaded.")
    for data in train_loader:
        pipe.scheduler.set_timesteps(args.num_inference_steps)
        
        #set the scheduler of GenHowTo (on per instance bases)
        if args.num_steps_to_skip is not None:  # possibly do not start from complete noise
            pipe.scheduler = DDIMSkipScheduler.from_config(pipe.scheduler.config)
            pipe.scheduler.set_num_steps_to_skip(args.num_steps_to_skip, args.num_inference_steps)
            print(f"Skipping first {args.num_steps_to_skip} DDIM steps, i.e., running DDIM from timestep "
                f"{pipe.scheduler.timesteps[0]} to {pipe.scheduler.timesteps[-1]}.")
            
        import pdb; pdb.set_trace()
        """
        # latents must be passed explicitly, otherwise the model generates incorrect shape
        latents = torch.randn((args.batch_size, 4, 64, 64))

        if args.num_inference_steps is not None:
            z = pipe.control_image_processor.preprocess(image)
            z = z * pipe.vae.config.scaling_factor
            t = pipe.scheduler.timesteps[0]
            alpha_bar = pipe.scheduler.alphas_cumprod[t].item()
            latents = math.sqrt(alpha_bar) * z + math.sqrt(1. - alpha_bar) * latents.to(z.device)

        output = pipe(
            args.prompt, image,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            latents=latents,
            num_images_per_prompt=args.num_images,
        ).images
        """

if __name__ == "__main__":
    args = create_parser()

    if args.dataset == 'crosstask':
        if args.split == 'base':
            from dataset.my_crosstask_dataloader import CrossTaskDataset as ProcedureDataset
        elif args.split == 'pdpp':
            # use PDPP data split and data sample
            from dataset.crosstask_dataloader_pdpp import CrossTaskDataset as ProcedureDataset
        elif args.split == 'p3iv':
            # use P3IV data split and data sample
            assert args.max_traj_len == 3, "Only the datasplit for max_traj_len = 3 is available."
            from dataset.crosstask_dataloader_p3iv import CrossTaskDataset as ProcedureDataset
    
    elif args.dataset == 'coin':
        from dataset.coin_dataloader import CoinDataset as ProcedureDataset
    
    elif args.dataset == 'niv':
        from dataset.niv_dataloader import NivDataset as ProcedureDataset

    run_genhowto(args)

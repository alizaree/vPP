import torch
import os
import time
import numpy as np
import math
from utils import *
from metrics import *
from torch.utils.data import DataLoader
from models.procedure_model import ProcedureModel
from models.utils import AverageMeter
import wandb
from tools.parser import create_parser
from PIL import Image
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.genhowto_model import MyPipe
from models.model import AutoregressiveTransformer
from tqdm import tqdm


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
    args.device=device
    if args.dataset == 'crosstask':
        logger.info("Loading prompt features...")
        file_path = './data/descriptors_crosstask.json'
        # Load the JSON file
        with open(file_path, 'r') as f:
            state_prompts = json.load(f)
        n_actions=sum([len(item) for key,item in state_prompts.items()])
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
    
    pipe = MyPipe(args.weights_path, args, device=device)
    logger.info("the model is loaded.")
    model= AutoregressiveTransformer(**vars(args)).to(device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)  # Adjust patience and factor as needed
    model.eval()

    # getting all_action_embeds and state embeds
    action_embeds_file = "all_action_embeds.pt"
    tstate_embeds_file = "all_tstate_embeds.pt"
    Load_from_save=True
    # Check if loading from save is required
    if Load_from_save:
        if os.path.exists(action_embeds_file) and os.path.exists(tstate_embeds_file):
            logger.info("Loading all_action_embeds and all_tstate_embeds from saved files...")
            all_action_embeds = torch.load(action_embeds_file)
            all_tstate_embeds = torch.load(tstate_embeds_file)
            logger.info("Loaded all_action_embeds and all_tstate_embeds from saved files.")
        else:
            logger.warning("No saved files found. Proceeding with estimation.")
            Load_from_save = False  # Set flag to False for estimation

    # If not loading from saved files or files not found, estimate tensors
    if not Load_from_save:
        logger.info("Estimating all the action and state prompt embeddings...")
        seen_actions = {}
        all_action_embeds = torch.zeros((n_actions, args.input_dim))  # Assuming n_actions and dim are known
        all_tstate_embeds = torch.zeros((n_actions, args.input_dim))  # Assuming n_actions and dim are known
        total_batches = n_actions
        with tqdm(total=total_batches) as pbar:
            for data in train_loader:
                pipe.set_timesteps(args.num_inference_steps)
                pipe.set_num_steps_to_skip(args.num_steps_to_skip, args.num_inference_steps)
                _, tstate_embeds, action_embds = pipe.extract_embeddings(data, pipe.model.tokenizer)
                for plan_id, plan in enumerate(data[2]):
                    for action_id, action in enumerate(plan):
                        action_val = action.item()
                        if action_val not in seen_actions:
                            all_action_embeds[action_val, :] =  action_embds[plan_id, action_id, :].clone()
                            all_tstate_embeds[action_val, :] =  tstate_embeds[plan_id, action_id, :].clone()
                            seen_actions[action_val] = 1
                            pbar.update(1)
                if len(seen_actions) == n_actions:
                    break

        # Save the tensors
        logger.info("Saving all_action_embeds and all_tstate_embeds...")
        torch.save(all_action_embeds, action_embeds_file)
        torch.save(all_tstate_embeds, tstate_embeds_file)
        logger.info("Saved all_action_embeds and all_tstate_embeds.")

    logger.info("Got all the action and state prompts embeds.")

    for epoch in range(args.epochs):
        # Set model to training mode
        model.train()
        # Initialize variables for tracking loss
        epoch_loss = 0.0
        for data in train_loader:
            pipe.set_timesteps(args.num_inference_steps)
            pipe.set_num_steps_to_skip(args.num_steps_to_skip, args.num_inference_steps)
            vis_embds, tstate_embds, action_embds = pipe.extract_embeddings(data, pipe.model.tokenizer)
            out_model, loss= model(vis_embds, action_embds)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate loss for the epoch
            epoch_loss += loss.item()
            
            import pdb; pdb.set_trace()
        # Average epoch loss
        avg_epoch_loss = epoch_loss/len(train_loader) 
        # Log the average loss for the epoch
        #wandb.log({"Training Loss": avg_epoch_loss}, step=epoch)

        # Adjust learning rate based on validation loss
        scheduler.step(avg_epoch_loss)
    
    # Finish logging
    #wandb.finish()


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

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
    wandb_config=vars(args)
    wandb.init(project='vPP', config=wandb_config)
    
    model = ProcedureModel(
        vis_input_dim=args.img_input_dim,
        lang_input_dim=args.text_input_dim,
        embed_dim=args.embed_dim,
        time_horz=args.max_traj_len, 
        num_classes=args.num_action,
        num_tasks=args.num_tasks,
        args=args
    ).to(device)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters()},
        ],
        lr=args.lr
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.step_size, 
        gamma=args.lr_decay, 
        last_epoch=-1
    )

    state_prompt_features  = torch.tensor(state_prompt_features).to(device, dtype=torch.float32).clone().detach()

    max_SR = 0

    for e in range(0, args.epochs):
        model.train()
        # losses
        losses_state  = AverageMeter()
        losses_action = AverageMeter()
        losses_task = AverageMeter()
        losses_state_pred = AverageMeter()

        # metrics for action
        action_acc1 = AverageMeter()
        action_acc5 = AverageMeter()
        action_sr   = AverageMeter()
        action_miou = AverageMeter()
        state_acc = AverageMeter()
        task_acc = AverageMeter()

        for i, (batch_states, batch_actions, batch_tasks) in enumerate(train_loader):
            '''
            batch_states:  (batch_size, time_horizon, 2, embedding_dim)
            batch_actions: (batch_size, time_horizon)
            '''
            batch_size, _ = batch_actions.shape
            optimizer.zero_grad()

            ## compute loss
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            batch_tasks = batch_tasks.to(device)
            outputs, labels, losses = model(
                visual_features=batch_states,
                state_prompt_features=state_prompt_features,
                actions=batch_actions,
                tasks=batch_tasks
            )

            total_loss = losses["action"] + losses["state_encode"] + losses["task"] + losses["state_decode"] * 0.1
            total_loss.backward()
            optimizer.step()
            
            losses_action.update(losses["action"].item())
            losses_state.update(losses["state_encode"].item())
            losses_task.update(losses["task"].item())
            losses_state_pred.update(losses["state_decode"].item())

            ## compute accuracy for state encoding
            acc_state = topk_accuracy(output=outputs["state_encode"].cpu(), target=labels["state"].cpu())
            state_acc.update(acc_state[0].item())

            ## compute accuracy for action prediction
            (acc1, acc5), sr, MIoU = \
                accuracy(outputs["action"].contiguous().view(-1, outputs["action"].shape[-1]).cpu(), 
                         labels["action"].contiguous().view(-1).cpu(), topk=(1, 5), max_traj_len=args.max_traj_len) 
            action_acc1.update(acc1.item())
            action_acc5.update(acc5.item())
            action_sr.update(sr.item())
            action_miou.update(MIoU)

            acc_task = topk_accuracy(output=outputs["task"].cpu(), target=labels["task"].cpu(), topk=[1])[0]
            task_acc.update(acc_task.item())

        logger.info("Epoch: {} State Loss: {:.2f} Top1 Acc: {:.2f}%"\
                    .format(e+1, losses_state.avg, state_acc.avg))
        logger.info("\tAction Loss: {:.2f}, SR: {:.2f}% Acc1: {:.2f}% Acc5: {:.2f}% MIoU: {:.2f}"\
                    .format(losses_action.avg,
                            action_sr.avg,
                            action_acc1.avg,
                            action_acc5.avg,
                            action_miou.avg))
        logger.info("\tTask Loss: {:.2f}, Acc1: {:.2f}%".format(losses_task.avg, task_acc.avg))
        logger.info("\tState Pred Loss: {:.2f}".format(losses_state_pred.avg))

        lr = optimizer.param_groups[0]['lr']
        ToLog={'lr/lr': lr, 'train_loss/state': losses_state.avg, 'train_loss/action': losses_action.avg,
                'train_loss/task': losses_task.avg, 'train_loss/state_pred': losses_state_pred.avg,
                'train_state/acc': state_acc.avg, 'train_action/sr': action_sr.avg,
                'train_action/miou': action_miou.avg, 'train_action/acc1': action_acc1.avg,
                'train_action/acc5': action_acc5.avg, 'train_task/acc': task_acc.avg}
        # Log scalars
        wandb.log(ToLog, step=e+1)


        if args.last_epoch < 0 or e < args.last_epoch:
            scheduler.step()

        ## validation
        if (e+1)%validate_interval == 0:
            model.eval()
            SR = eval(args, 
                      valid_loader, 
                      model, 
                      logger, 
                      state_prompt_features, 
                      transition_matrix, 
                      e, 
                      device,
                      writer=writer, 
                      is_train=True)
            
            # save the last model to logger path
            torch.save(
                model.state_dict(), 
                os.path.join(
                    logger_path,
                    f"T{args.max_traj_len}_model_last.pth"  
                )
            )
            # save the best model to checkpoints path
            if SR > max_SR:
                max_SR = SR
                log_save_path = os.path.join(
                    logger_path,
                    f"T{args.max_traj_len}_model_best.pth"  
                )
                checkpoint_save_path = os.path.join(
                        args.saved_path, 
                        args.dataset,
                        f"T{args.max_traj_len}_model_best.pth"
                    )
                torch.save(model.state_dict(), checkpoint_save_path)        
                os.system(f"cp {checkpoint_save_path} {log_save_path}")

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

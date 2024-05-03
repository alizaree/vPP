import torch
import numpy as np
from utils import *
from metrics import *
from models.utils import AverageMeter

from PIL import Image

from models.GenHowTo.genhowto_utils import load_genhowto_model, DDIMSkipScheduler

class MyPipe:
    def __init__(self, weights_path, args, device):
        self.model = load_genhowto_model(weights_path, device)
        self.device = device
        self.args=args
    def set_timesteps(self, num_inference_steps):
        self.model.scheduler.set_timesteps(num_inference_steps)
    
    def set_num_steps_to_skip(self, num_steps_to_skip, num_inference_steps):
        if num_steps_to_skip is not None:
            self.model.scheduler = DDIMSkipScheduler.from_config(self.model.scheduler.config)
            self.model.scheduler.set_num_steps_to_skip(num_steps_to_skip, num_inference_steps)
            print(f"Skipping first {num_steps_to_skip} DDIM steps, i.e., running DDIM from timestep "
                  f"{self.model.scheduler.timesteps[0]} to {self.model.scheduler.timesteps[-1]}.")

    def preprocess_images(self, image_data):
        img_input = [Image.fromarray((idd.numpy()).astype(np.uint8)) for idd in image_data]
        with torch.no_grad():
            vae_out = self.model.control_image_processor.preprocess(img_input)
        return vae_out

    def encode_tstate(self, prompt_s, tokenizer, batch_size=None):
        if batch_size==None:
            batch_size=self.args.batch_size
        with torch.no_grad():
            tokens = tokenizer(prompt_s, padding=True, return_tensors='pt')
            input_ids = tokens['input_ids'].to(self.device)
            last_hidden_states = self.model.text_encoder(input_ids)['last_hidden_state']
            eos_positions = (input_ids == tokenizer.eos_token_id).nonzero()
            tstate_embds = last_hidden_states[eos_positions[:, 0], eos_positions[:, 1]].view(
                batch_size, 1, -1)
        return tstate_embds

    def encode_actions(self, prompt_a, tokenizer, batch_size=None):
        if batch_size==None:
            batch_size=self.args.batch_size
        
        with torch.no_grad():
            tokens = tokenizer(prompt_a, padding=True, return_tensors='pt')
            input_ids = tokens['input_ids'].to(self.device)
            last_hidden_states = self.model.text_encoder(input_ids)['last_hidden_state']
            eos_positions = (input_ids == tokenizer.eos_token_id).nonzero()
            action_embds = last_hidden_states[eos_positions[:, 0], eos_positions[:, 1]].view(
                batch_size, 1, -1)
        return action_embds

    def extract_embeddings(self, data, tokenizer):
        # Extract visual embeddings
        reshaped_tensor = data[0].view(-1, *data[0].shape[3:])
        vae_out = self.preprocess_images(reshaped_tensor)
        vis_embds = vae_out.view(data[0].shape[0], data[0].shape[1], 2, *vae_out.shape[1:])

        # Extract text embeddings for states
        state_mode = 'after'
        prompt_s = [data[1][ac_id][state_mode][0][batch_id] for batch_id in range(self.args.batch_size) for ac_id in
                    range(len(data[1]))]
        tstate_embds = self.encode_tstate(prompt_s, tokenizer)

        # Extract text embeddings for actions
        state_mode = 'description'
        prompt_a = [data[1][ac_id][state_mode][batch_id] for batch_id in range(self.args.batch_size) for ac_id in
                    range(len(data[1]))]
        action_embds = self.encode_actions(prompt_a, tokenizer)

        return vis_embds, tstate_embds, action_embds
    
    def extract_embeddings_inDL(self, data, tokenizer): # the class to extract embedding directly in DataLoader class
        # Extract visual embeddings
        
        reshaped_tensor = data[0].view(-1, *data[0].shape[2:])
        vae_out = self.preprocess_images(reshaped_tensor)
        vis_embds = vae_out.view(data[0].shape[0], 2, *vae_out.shape[1:])
        
            

        # Extract text embeddings for states
        state_mode = 'after'
        prompt_s = [data[1][ac_id][state_mode][0] for ac_id in range(len(data[1]))]
        tstate_embds = self.encode_tstate(prompt_s, tokenizer, batch_size=len(prompt_s))

        # Extract text embeddings for actions
        state_mode = 'description'
        prompt_a = [data[1][ac_id][state_mode] for ac_id in range(len(data[1]))]
        action_embds = self.encode_actions(prompt_a, tokenizer, batch_size=len(prompt_a))
        return vis_embds, tstate_embds, action_embds
import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
import json
import os
import cv2
from tqdm import tqdm
from PIL import Image
from models.genhowto_model import MyPipe

class CrossTaskDataset(Dataset):
    def __init__(
        self, 
        anot_info,
        img_dir,
        embedding_dir,
        prompt_json, 
        video_list, 
        horizon = 3, 
        num_action = 133, 
        aug_range = 0, 
        M = 2, 
        mode = "train",
        vid_dir="",
        save_image_states=False,
        save_embeddings=False,
        args=None,
    ):
        super().__init__()
        self.anot_info = anot_info
        self.vid_dir= vid_dir
        self.img_dir = img_dir
        self.embedding_dir=embedding_dir
        #only for google cloud:
        
        #self.storage_client = storage.Client()
        #self.bucket_img = self.storage_client.bucket(self.img_dir)
        
        self.prompt_json = prompt_json
        self.aug_range = aug_range
        self.horizon = horizon
        self.video_list = video_list
        self.max_duration = 0
        self.mode = mode
        self.M = M
        self.num_action = num_action
        self.args=args
        self.transition_matrix = np.zeros((num_action, num_action))
        self.task_info = {"Make Jello Shots": 23521, 
                          "Build Simple Floating Shelves": 59684, 
                          "Make Taco Salad": 71781, 
                          "Grill Steak": 113766,
                          "Make Kimchi Fried Rice": 105222, 
                          "Make Meringue": 94276,
                          "Make a Latte": 53193, 
                          "Make Bread and Butter Pickles": 105253,
                          "Make Lemonade": 44047, 
                          "Make French Toast": 76400,
                          "Jack Up a Car": 16815, 
                          "Make Kerala Fish Curry": 95603,
                          "Make Banana Ice Cream": 109972, 
                          "Add Oil to Your Car": 44789,
                          "Change a Tire": 40567, 
                          "Make Irish Coffee": 77721,
                          "Make French Strawberry Cake": 87706, 
                          "Make Pancakes": 91515}

        self.data = []
        if save_image_states:
            self.SaveImageStates()
        if save_embeddings:
            self.pipe= MyPipe(args.weights_path, args, device=args.device)
            self.save_embeds()
        self.load_data()
        if self.mode == "train":
            self.transition_matrix = self.cal_transition(self.transition_matrix)

    def cal_transition(self, matrix):
        ''' Cauculate transition matrix

        Args:
            matrix:     [num_action, num_action]

        Returns:
            transition: [num_action, num_action]
        '''
        transition = matrix / np.sum(matrix, axis = 1, keepdims = True)
        return transition
    def load_npy_gcloud(self, file_name):
        
        blob = self.bucket_img.blob(file_name)
        with blob.open('rb') as f:
                data = np.load(f,allow_pickle=True)
        return data.item()
    def fix_image_size(self, image):
        # Convert numpy array to PIL Image
        image = Image.fromarray(image)

        w, h = image.size
        if w > h:
            image = image.crop(((w - h) // 2, 0, (w + h) // 2, h))
        elif h > w:
            image = image.crop((0, (h - w) // 2, w, (h + w) // 2))
        image = image.resize((512, 512))

        # Convert PIL Image back to numpy array
        image = np.array(image)
        return image
        

    def SaveImageStates(self):
        with open(self.video_list, "r") as f:
            video_info_dict = json.load(f)
        
        for video_info in tqdm(video_info_dict, desc="Processing videos"):
            frame_list_start=[]
            frame_list_end=[]
            video_id = video_info["id"]["vid"]
            if os.path.exists(os.path.join(self.img_dir,video_id+".npy")):
                continue
            video_anot = self.anot_info[video_id]
            video_path = os.path.join(self.vid_dir, f"{video_id}.mp4")
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                for action_info in video_anot:
                    start_frame_index = int(action_info["start"] * fps)
                    end_frame_index = int(action_info["end"] * fps)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_index)
                    ret, start_frame = cap.read()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, end_frame_index)
                    ret, end_frame = cap.read()
                    # make RGB
                    start_frame=self.fix_image_size(cv2.cvtColor(start_frame, cv2.COLOR_BGR2RGB))
                    end_frame=self.fix_image_size(cv2.cvtColor(end_frame, cv2.COLOR_BGR2RGB))
                    
                    if ret:
                        frame_list_start.append(start_frame[None, :])
                        frame_list_end.append(end_frame[None,:])
                
                if len(frame_list_start)!=len(video_anot) or len(frame_list_end)!=len(video_anot) :
                    cap.release()
                    continue
                all_data = {
                "video_id": video_id,
                "start_frames": np.concatenate(frame_list_start, axis=0) ,
                "end_frames": np.concatenate(frame_list_end, axis=0) ,
                "action_ids": video_anot,
                }
                cap.release()
                # Save the dictionary as a .npy file
                np.save(os.path.join(self.img_dir,video_id+".npy"), all_data)
            except:
                cap.release()
                continue
            
    def save_embeds(self,):
        with open(self.video_list, "r") as f:
            video_info_dict = json.load(f)
        
        for video_info in tqdm(video_info_dict, desc="Processing video embeddings"):
            video_id = video_info["id"]["vid"]
            video_anot = self.anot_info[video_id]
            task_id = video_anot[0]["task_id"]
            task = video_anot[0]["task"]
            
            
            try:
                # Check if the embeddings already exist
                if (os.path.exists(os.path.join(self.embedding_dir, f"{video_id}_visual_embeddings.pt")) and
                    os.path.exists(os.path.join(self.embedding_dir, f"{video_id}_text_state_embeddings.pt")) and
                    os.path.exists(os.path.join(self.embedding_dir, f"{video_id}_action_embeddings.pt"))):
                    continue
               
                #frame_data=self.load_npy_gcloud("{}.npy".format(video_id))
                frame_data= np.load(os.path.join(self.img_dir, "{}.npy".format(video_id)), allow_pickle=True).item()
                start_frames = frame_data["start_frames"]
                end_frames = frame_data["end_frames"]
                Frames=np.concatenate((start_frames[:,None,...], end_frames[:,None,...]),axis=1)
                Prompts=[]
                for cur_video_anot in video_anot:
                    Prompts.append(self.prompt_json[cur_video_anot["task"]][cur_video_anot["action"]])
                # processing large files in chuncks for memory limits
                m=10
                num_chunks = len(Frames) // m
                if len(Frames) % m != 0:
                    num_chunks += 1
                vis_embds=[]
                tstate_embds=[]
                action_embds=[]
                for i in range(num_chunks):
                    start_idx = i * m
                    end_idx = min((i + 1) * m, len(Frames))
                    chunk_data = (torch.tensor(Frames[start_idx:end_idx]), Prompts[start_idx:end_idx])
                    with torch.no_grad():
                        self.pipe.set_timesteps(self.args.num_inference_steps)
                        self.pipe.set_num_steps_to_skip(self.args.num_steps_to_skip, self.args.num_inference_steps)
                        vis_embds_chunk, tstate_embds_chunk, action_embds_chunk = self.pipe.extract_embeddings_inDL(chunk_data, self.pipe.model.tokenizer)
                    vis_embds.append(vis_embds_chunk.detach().cpu())
                    tstate_embds.append(tstate_embds_chunk.detach().cpu())
                    action_embds.append(action_embds_chunk.detach().cpu())
                    del vis_embds_chunk 
                    del tstate_embds_chunk
                    del action_embds_chunk
                vis_embds=torch.cat(vis_embds, dim=0)
                tstate_embds=torch.cat(tstate_embds,dim=1)
                action_embds=torch.cat(action_embds,dim=1)
                # Save embeddings using torch.save()
                torch.save(vis_embds, os.path.join(self.embedding_dir,video_id+'_visual_embeddings.pt'))
                torch.save(tstate_embds, os.path.join(self.embedding_dir,video_id+'_text_state_embeddings.pt'))
                torch.save(action_embds, os.path.join(self.embedding_dir,video_id+'_action_embeddings.pt'))
            except:
                #import pdb; pdb.set_trace()
                continue
        
            
    def load_data(self,):
        with open(self.video_list, "r") as f:
            video_info_dict = json.load(f)
        
        for video_info in tqdm(video_info_dict, desc="Processing videos"):
            video_id = video_info["id"]["vid"]
            video_anot = self.anot_info[video_id]
            task_id = video_anot[0]["task_id"]
            task = video_anot[0]["task"]
            
            try:
                frame_data= np.load(os.path.join(self.img_dir, "{}.npy".format(video_id)), allow_pickle=True).item()
                start_frames = frame_data["start_frames"]
                end_frames = frame_data["end_frames"]
                vis_embds_loaded = torch.load(os.path.join(self.embedding_dir,video_id+'_visual_embeddings.pt')) # n_actions, 2, 4,64,64
                tstate_embds_loaded = torch.load(os.path.join(self.embedding_dir,video_id+'_text_state_embeddings.pt')) # 1, n_actions, dim
                action_embds_loaded = torch.load(os.path.join(self.embedding_dir,video_id+'_action_embeddings.pt')) # 1, n_actions, dim
                
            except:
                continue
                        
            # Remove repeated actions. Intuitively correct, but do not work well on dataset.
            # legal_video_anot = []
            # for i in range(len(video_anot)):
            #     if i == 0 or video_anot[i]["action_id"] != video_anot[i-1]["action_id"]:
            #         legal_video_anot.append(video_anot[i])
            # video_anot = legal_video_anot

            ## update transition matrix
            if self.mode == "train":
                for i in range(len(video_anot)-1):
                    cur_action = video_anot[i]["action_id"]-1
                    next_action = video_anot[i+1]["action_id"]-1
                    self.transition_matrix[cur_action, next_action] += 1


            for i in range(len(video_anot)-self.horizon+1):
                all_action_ids = []
                all_frames = []
                all_prompts= []
                all_v_embeds=[]
                all_tstate_embeds=[]
                all_action_embeds=[]
                for j in range(self.horizon):
                    cur_video_anot = video_anot[i+j]
                    cur_action_id = cur_video_anot["action_id"]-1
                    cur_prompts=self.prompt_json[cur_video_anot["task"]][cur_video_anot["action"]]
                    
                    all_prompts.append(cur_prompts)
                    all_action_ids.append(cur_action_id)
                        
                    s_frame=start_frames[i+j,...]
                    e_frame=end_frames[i+j,...]
                    v_embd= vis_embds_loaded[i+j,...]
                    tstate_embd= tstate_embds_loaded[0,i+j,...]
                    action_embd= action_embds_loaded[0,i+j,...]
                    
                    all_frames.append(np.stack((s_frame, e_frame)))
                    all_v_embeds.append(v_embd)
                    all_tstate_embeds.append(tstate_embd)
                    all_action_embeds.append(action_embd)
                task_id = cur_video_anot["task_id"]

                ## permutation of frames, action ids and prompts
                #frames_per = itertools.product(*all_frames)

                self.data.extend([{"frames": np.stack(all_frames),
                                   "actions": np.array(all_action_ids),
                                   "tasks": np.array(task_id),
                                   "prompts":  all_prompts,
                                   "vis_embeds": torch.stack(all_v_embeds),
                                   "tstate_embeds": torch.stack(all_tstate_embeds),
                                   "action_embeds": torch.stack(all_action_embeds)}])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        actions = self.data[idx]["actions"]
        tasks = self.data[idx]["tasks"]
        frames = np.squeeze(self.data[idx]["frames"])
        prompts = self.data[idx]["prompts"]
        vis_embeds= self.data[idx]["vis_embeds"]
        tstate_embeds= self.data[idx]["tstate_embeds"]
        action_embeds= self.data[idx]["action_embeds"]
        return torch.as_tensor(vis_embeds, dtype=torch.float32), torch.as_tensor(tstate_embeds, dtype=torch.float32), torch.as_tensor(action_embeds, dtype=torch.float32), torch.as_tensor(frames, dtype=torch.float32), prompts, torch.as_tensor(actions, dtype=torch.long), torch.as_tensor(tasks, dtype=torch.long)

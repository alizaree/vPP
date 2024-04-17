import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
import json
import os
import cv2
from tqdm import tqdm
from google.cloud import storage

class CrossTaskDataset(Dataset):
    def __init__(
        self, 
        anot_info,
        img_dir,
        prompt_json, 
        video_list, 
        horizon = 3, 
        num_action = 133, 
        aug_range = 0, 
        M = 2, 
        mode = "train",
        vid_dir="",
        save_image_states=False,
    ):
        super().__init__()
        self.anot_info = anot_info
        self.vid_dir= vid_dir
        self.img_dir = img_dir
        #only for google cloud:
        
        self.storage_client = storage.Client()
        self.bucket_img = self.storage_client.bucket(self.img_dir)
        
        self.prompt_json = prompt_json
        self.aug_range = aug_range
        self.horizon = horizon
        self.video_list = video_list
        self.max_duration = 0
        self.mode = mode
        self.M = M
        self.num_action = num_action
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
        
        storage_client = storage.Client()
        blob = self.bucket_img.blob(file_name)
        with blob.open('rb') as f:
                data = np.load(f,allow_pickle=True)
        return data.item()
        

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
        
            
    def load_data(self):
        with open(self.video_list, "r") as f:
            video_info_dict = json.load(f)
        
        for video_info in tqdm(video_info_dict, desc="Processing videos"):
            video_id = video_info["id"]["vid"]
            video_anot = self.anot_info[video_id]
            task_id = video_anot[0]["task_id"]
            task = video_anot[0]["task"]
            
            try:
               
                frame_data=self.load_npy_gcloud("{}.npy".format(video_id))
                #np.load(os.path.join(self.img_dir, "{}.npy".format(video_id)), allow_pickle=True).item()
                start_frames = frame_data["start_frames"]
                end_frames = frame_data["end_frames"]
                
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
                
                for j in range(self.horizon):
                    cur_video_anot = video_anot[i+j]
                    cur_action_id = cur_video_anot["action_id"]-1
                    
                    all_action_ids.append(cur_action_id)
                        
                    s_frame=cv2.cvtColor(start_frames[i+j,...], cv2.COLOR_BGR2RGB)
                    e_frame=cv2.cvtColor(end_frames[i+j,...], cv2.COLOR_BGR2RGB)
                    all_frames.append([np.stack((s_frame, e_frame))])

                task_id = cur_video_anot["task_id"]

                ## permutation of frames, action ids and prompts
                frames_per = itertools.product(*all_frames)

                self.data.extend([{"frames": np.stack(f),
                                   "actions": np.array(all_action_ids),
                                   "tasks": np.array(task_id)} 
                                  for f in frames_per])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        actions = self.data[idx]["actions"]
        tasks = self.data[idx]["tasks"]
        frames = np.squeeze(self.data[idx]["frames"])
        return torch.as_tensor(states, dtype=torch.float32), torch.as_tensor(actions, dtype=torch.long), torch.as_tensor(tasks, dtype=torch.long), torch.as_tensor(frames, dtype=torch.float32)

import numpy as np
import itertools
from torch.utils.data import Dataset
import torch
import json
import os
import cv2
from tqdm import tqdm
from PIL import Image
import PIL.PngImagePlugin
from models.genhowto_model import MyPipe
from transformers import AutoProcessor, Blip2ForConditionalGeneration,  InstructBlipProcessor, InstructBlipForConditionalGeneration
import google.generativeai as genai
from google.cloud import storage
import asyncio
from transformers import CLIPProcessor, CLIPModel

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

        self.norm_stat=False
        if args.normalize_features:
            print("Embeddings will be normalized.")
            self.assign_stat()
        
        self.data = []
        if save_image_states:
            self.SaveImageStates()

        if args.save_captions:
            if args.cap_model=="BLIP2":
                self.cap_processor = AutoProcessor.from_pretrained(args.cap_model_checkpoint)
                self.cap_model = Blip2ForConditionalGeneration.from_pretrained(args.cap_model_checkpoint, torch_dtype=torch.float16).to(args.device)
            elif args.cap_model=="Gemeni":
                with open(args.cap_model_key_path, 'r') as file:
                    content = file.read()
                GOOGLE_API_KEY=content
                genai.configure(api_key=GOOGLE_API_KEY)
                self.cap_model = genai.GenerativeModel('gemini-pro-vision')
            elif args.cap_model=="InstructBlip":
                self.cap_model=InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b").to(args.device)
                self.cap_processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")    
            self.save_captions()
        
        if save_embeddings:
            if self.args.feature_extractor=="GenHowTo":
                self.pipe= MyPipe(args.weights_path, args, device=args.device)
            elif self.args.feature_extractor=="CLIP":
                # Load the CLIP model
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)
                # Load the CLIP processor
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            else:
                raise NotImplementedError
            self.save_embeds()
        
        self.load_data()
        if self.mode == "train":
            self.transition_matrix = self.cal_transition(self.transition_matrix)
    def assign_stat(self,):
        self.norm_stat=True
        self.state_mean=0.0134 #-0.1706
        self.state_std=0.4442 #1.0679
        self.action_mean=0.0128 #-0.1696
        self.action_std=0.4225 #1.0706
        self.mean_vis=-0.0098 # 0.6082
        self.std_vis=0.4686 #5.4927
            
    def cal_transition(self, matrix):
        ''' Cauculate transition matrix

        Args:
            matrix:     [num_action, num_action]

        Returns:
            transition: [num_action, num_action]
        '''
        transition = matrix / np.sum(matrix, axis = 1, keepdims = True)
        return transition
    
    async def process_images(self, prompt, image_list):
        tasks = [self.process_image(prompt, img) for img in image_list]
        results = await asyncio.gather(*tasks)
        return results
    
    async def process_image(self, prompt, img):
        try:
            r = await self.cap_model.generate_content_async([prompt, img])
            return r.text
        except Exception as e:
            return f"Error processing image: {str(e)}"
    
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
            
    def preprocess_images_CLIP(self, images):
        # Preprocess image using CLIP processor
        inputs = self.clip_processor(images=images, return_tensors="pt").to(self.args.device)
        # Extract image features
        with torch.no_grad():
            image_features= self.clip_model.get_image_features(**inputs)
            
        return image_features

    def encode_tstate_CLIP(self, prompts):
        text_features = []
        for prompt in prompts:
            # Preprocess text using CLIP processor
            inputs = self.clip_processor(text=prompt, return_tensors="pt", padding=True).to(self.args.device)
            # Extract text features
            with torch.no_grad():
                text_features.append(self.clip_model.get_text_features(**inputs))
        return torch.stack(text_features)
            
    def extract_embeddings_CLIP(self, data):
        # Extract visual embeddings
        reshaped_tensor = data[0].view(-1, *data[0].shape[2:])
        vis_embds = self.preprocess_images_CLIP(reshaped_tensor)
        vis_embds = vis_embds.view(data[0].shape[0], data[0].shape[1], *vis_embds.shape[1:])

        # Extract text embeddings for states
        state_mode = 'after'
        prompt_s = [data[1][ac_id][state_mode][0] for ac_id in range(len(data[1]))]
        tstate_embds = self.encode_tstate_CLIP(prompt_s)

        # Extract text embeddings for actions
        state_mode = 'description'
        prompt_a = [data[1][ac_id][state_mode] for ac_id in range(len(data[1]))]
        action_embds = self.encode_tstate_CLIP(prompt_a)
        return vis_embds, tstate_embds, action_embds
            
    def save_captions(self,):
        with open(self.video_list, "r") as f:
            video_info_dict = json.load(f)
        
        for video_info in tqdm(video_info_dict, desc="Processing video embeddings"):
            video_id = video_info["id"]["vid"]
            video_anot = self.anot_info[video_id]
            task_id = video_anot[0]["task_id"]
            task = video_anot[0]["task"]
            save_path=os.path.join(self.args.cap_dir,video_id+'_frame_captions.txt')
            
            
            try:
                # Check if the embeddings already exist
                if (os.path.exists(save_path)):
                    continue
               
                #frame_data=self.load_npy_gcloud("{}.npy".format(video_id))
                frame_data= np.load(os.path.join(self.img_dir, "{}.npy".format(video_id)), allow_pickle=True).item()
                start_frames = frame_data["start_frames"]
                end_frames = frame_data["end_frames"]
                if self.args.cap_model=="Gemeni":
                    prompt=f"Describe the content of image, related to the task of \"{task}\", in one sentence."
                    #get the start state descriptions
                    img_input_s= [Image.fromarray(idd,'RGB') for idd in start_frames]
                    img_input_e = [Image.fromarray(idd,'RGB') for idd in end_frames]
                    substring='Error processing image'
                    flag_rerun=True
                    while flag_rerun:
                        try:
                            response_start=[self.cap_model.generate_content([prompt, img]).text for img in img_input_s]
                            response_end=[self.cap_model.generate_content([prompt, img]).text for img in img_input_e]
                            flag_rerun= any(substring in item for item in response_start) or any(substring in item for item in response_end)
                        except:
                            flag_rerun=True
                    #try:
                    #    substring='Error processing image'
                    #    flag_rerun=True
                    #    while flag_rerun:
                    #        loop = asyncio.get_event_loop()
                    #        response_start = loop.run_until_complete(self.process_images(prompt, img_input_s))
                    #        loop = asyncio.get_event_loop()
                    #        response_end = loop.run_until_complete(self.process_images(prompt, img_input_e))
                    #        flag_rerun= any(substring in item for item in response_start) or any(substring in item for item in response_end)
                    #except:
                    #    import pdb; pdb.set_trace()
                elif self.args.cap_model=="InstructBlip":
                    prompt=f"Describe the content of image, related to the task of \"{task}\", in one sentence."
                    prompts=[prompt for _ in range(start_frames.shape[0])]
                    inputs = self.cap_processor(start_frames, text=prompts, return_tensors="pt").to(self.args.device, torch.float16)
                    outputs = self.cap_model.generate(
                            **inputs,
                            do_sample=False,
                            num_beams=5,
                            max_length=256,
                            min_length=1,
                            top_p=0.9,
                            repetition_penalty=1.5,
                            length_penalty=1.0,
                            temperature=1,
                        )
                    generated_text = self.cap_processor.batch_decode(outputs, skip_special_tokens=True)
                    raise NotImplementedError
                else:
                    raise NotImplementedError
                        
                Frames=list(zip(response_start,response_end ))
                with open(save_path, 'w') as file:
                        for frame in Frames:
                            file.write(f"{frame[0]};_,_,_;{frame[1]}\n")
            except:
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
                if (os.path.exists(os.path.join(self.embedding_dir, f"{video_id}_frame_embeddings.pt")) and
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
                        if self.args.feature_extractor=="GenHowTo":
                            self.pipe.set_timesteps(self.args.num_inference_steps)
                            self.pipe.set_num_steps_to_skip(self.args.num_steps_to_skip, self.args.num_inference_steps)
                            vis_embds_chunk, tstate_embds_chunk, action_embds_chunk = self.pipe.extract_embeddings_inDL(chunk_data, self.pipe.model.tokenizer)
                        elif self.args.feature_extractor=="CLIP":
                            vis_embds_chunk, tstate_embds_chunk, action_embds_chunk = self.extract_embeddings_CLIP(chunk_data)
                            
                    vis_embds.append(vis_embds_chunk.detach().cpu())
                    tstate_embds.append(tstate_embds_chunk.detach().cpu())
                    action_embds.append(action_embds_chunk.detach().cpu())
                    del vis_embds_chunk 
                    del tstate_embds_chunk
                    del action_embds_chunk
                vis_embds=torch.cat(vis_embds, dim=0)
                tstate_embds=torch.cat(tstate_embds,dim=0)
                action_embds=torch.cat(action_embds,dim=0)
                # Save embeddings using torch.save()
                torch.save(vis_embds, os.path.join(self.embedding_dir,video_id+'_frame_embeddings.pt'))
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
                vis_embds_loaded = torch.load(os.path.join(self.embedding_dir,video_id+'_frame_embeddings.pt')) # n_actions, 2, 4,64,64
                tstate_embds_loaded = torch.load(os.path.join(self.embedding_dir,video_id+'_text_state_embeddings.pt')) # 1, n_actions, dim
                action_embds_loaded = torch.load(os.path.join(self.embedding_dir,video_id+'_action_embeddings.pt')) # 1, n_actions, dim
                # Normalize if normalization is specified
                if self.norm_stat:
                    vis_embds_loaded=(vis_embds_loaded-self.mean_vis)/self.std_vis
                    tstate_embds_loaded= (tstate_embds_loaded- self.state_mean)/self.state_std
                    action_embds_loaded= (action_embds_loaded-self.action_mean)/self.action_std
                
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
                    tstate_embd= tstate_embds_loaded[i+j,...]
                    action_embd= action_embds_loaded[i+j,...]
                    
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

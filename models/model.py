import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from torch.nn import functional as F

class AutoregressiveTransformer(nn.Module):
    def __init__(self, d_model=512, max_traj_len=3, device='cpu', input_dim=512, num_action=133, vis_input_dim=64, dropout_rate=0.1, n_layer=4, n_head=8, **args):
        super(AutoregressiveTransformer, self).__init__()

        self.d_model = d_model
        self.max_traj_len = max_traj_len
        self.vis_input_dim = vis_input_dim
        self.device = device

        # Learnable positional embeddings
        self.positional_embeddings = torch.nn.Embedding(max_traj_len, d_model) 
        
        transformer_configs=GPT2Config()
        transformer_configs.n_positions=self.max_traj_len+2
        transformer_configs.n_ctx=transformer_configs.n_positions
        #### model_dims
        transformer_configs.n_embd=d_model
        transformer_configs.n_layer=n_layer
        transformer_configs.n_head=n_head
        transformer_configs.resid_pdrop=dropout_rate
        transformer_configs.embd_pdrop=dropout_rate
        transformer_configs.attn_pdrop=dropout_rate
        transformer_configs.vocab_size=num_action +100
        self.transformer_configs=transformer_configs

        # Transformer model
        self.transformer = GPT2Model(self.transformer_configs).to(device)

        # Projection layer
        self.projection_layer = nn.Sequential(
            nn.Linear(d_model, 2*d_model),  #
            nn.ReLU(),
            nn.Linear(2*d_model, 2*d_model),  # Output shape: n_batch, d_model
            nn.ReLU(),
            nn.Linear(2*d_model, d_model),  #
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # MLP for encoding state
        #self.state_encoder = nn.Sequential(
        #    nn.Conv2d(4, 1, kernel_size=3, stride=2, padding=1),  # Change input channels to 4
        #    nn.ReLU(),
        #    nn.Flatten(),
        #    nn.Linear(32 * 32, d_model),  # Output shape: n_batch, d_model
        #    nn.ReLU(),
        #    nn.Dropout(dropout_rate)
        #)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, 2*d_model),  #
            nn.ReLU(),
            nn.Linear(2*d_model, 2*d_model),  # Output shape: n_batch, d_model
            nn.ReLU(),
            nn.Linear(2*d_model, d_model),  #
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Additional layers can be added if necessary
        # Classifier for action prediction
        self.action_classifier  = nn.Sequential(
            nn.Linear(d_model, d_model),  #
            nn.ReLU(),
            nn.Linear(d_model,num_action),  # Output shape: n_batch, d_model
            nn.ReLU(),
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward_run(self, visual_input, ground_truth_action_indices=None, ground_truth_action_embeds=None, all_action_embeds=None,  loss='mse'):
        #n_batch, n_action, 2, w, h, 3
        batch_size = visual_input.shape[0]
        visual_s=visual_input[:, 0,0,...]
        visual_f=visual_input[:,-1,-1,...]
        # Pass visual input through state encoder
        vis_s = self.state_encoder(visual_s)
        vis_f = self.state_encoder(visual_f)
        #get the initial learnble query: 
        idd=torch.LongTensor([0]).to(self.device)
        learnble_query = self.positional_embeddings(idd).expand(batch_size, -1)
        vis_in=torch.cat((vis_s[:,None,:],vis_f[:,None,:], learnble_query[:,None,:]),dim=1)
        transformer_output = self.transformer(inputs_embeds=vis_in).last_hidden_state[:,-1,:].clone()
        out=[transformer_output[:,None,:]]
        # Iterate through each action
        for i in range(1, self.max_traj_len):
            projected_output = self.projection_layer(transformer_output[:,None,:])
            idd=torch.LongTensor([i]).to(self.device)
            # Get learnable positional embedding for the current action
            learnble_query = self.positional_embeddings(idd).expand(batch_size, -1)[:,None,:]
            input_embed = projected_output + learnble_query
            vis_in =torch.cat((vis_in, input_embed),dim=1)
            transformer_output = self.transformer(inputs_embeds=vis_in).last_hidden_state[:,-1,:].clone()
            out.append(transformer_output[:,None,:])
        action_embeds=torch.cat(out,dim=1)
        # Compute MSE loss between predicted action embeddings and ground truth action embeddings
        if ground_truth_action_embeds!=None and loss=='mse':
            assert all_action_embeds!=None, "Provide all_action_embeds for finding indices under mse mode."
            #to do: calculate the loss
            mse_loss = F.mse_loss(action_embeds, ground_truth_action_embeds)
            predicted_indices = torch.argmax(torch.matmul(action_embeds.view(-1, action_embeds.size(-1)), all_action_embeds.T), dim=1).view(action_embeds.size(0), action_embeds.size(1))
            return action_embeds, mse_loss, predicted_indices
        elif loss=='ce' and ground_truth_action_indices!=None:
            # Predict action indices
            action_logits = self.action_classifier(action_embeds)

            # Reshape ground truth for loss calculation
            ground_truth_action_indices = ground_truth_action_indices.view(-1)

            # Calculate loss
            loss = self.loss_fn(action_logits.view(-1, action_logits.size(-1)), ground_truth_action_indices)
            probabilities = F.softmax(action_logits, dim=1)

            # Get the predicted class index for each sample
            predicted_classes = torch.argmax(probabilities, dim=-1)
            return action_embeds, loss, predicted_classes
        else:
            
            return action_embeds
        
    def forward(self, visual_input, ground_truth_action_indices=None, ground_truth_action_embeds=None, all_action_embeds=None, loss='mse'):
        #n_batch, n_action, 2, w, h, 3
        batch_size = visual_input.shape[0]
        visual_s=visual_input[:, 0,0,...]
        visual_f=visual_input[:,-1,-1,...]
        # Pass visual input through state encoder
        vis_s = self.state_encoder(visual_s)
        vis_f = self.state_encoder(visual_f)
        # get all the learnble queries
        learnble_query = torch.cat([self.positional_embeddings(torch.LongTensor([idd]).to(self.device)).expand(batch_size, -1)[:,None,:]
                          for idd in range(1,self.max_traj_len )], dim=1)
        in_act_embeds= self.projection_layer(ground_truth_action_embeds[:, :-1, ...]) + learnble_query
        #get the initial learnble query: 
        idd=torch.LongTensor([0]).to(self.device)
        learnble_query = self.positional_embeddings(idd).expand(batch_size, -1)
        vis_in=torch.cat((vis_s[:,None,:],vis_f[:,None,:], learnble_query[:,None,:], in_act_embeds),dim=1)
        action_embeds = self.transformer(inputs_embeds=vis_in).last_hidden_state[:,-3:,:].clone()
        
        # Compute MSE loss between predicted action embeddings and ground truth action embeddings
        if ground_truth_action_embeds!=None and loss=='mse':
            assert all_action_embeds!=None, "Provide all_action_embeds for finding indices under mse mode."
            #to do: calculate the loss
            mse_loss = F.mse_loss(action_embeds, ground_truth_action_embeds)
            predicted_indices = torch.argmax(torch.matmul(action_embeds.view(-1, action_embeds.size(-1)), all_action_embeds.T), dim=1).view(action_embeds.size(0), action_embeds.size(1))
            return action_embeds, mse_loss, predicted_indices
        elif loss=='ce' and ground_truth_action_indices!=None:
            # Predict action indices
            action_logits = self.action_classifier(action_embeds)

            # Reshape ground truth for loss calculation
            ground_truth_action_indices = ground_truth_action_indices.view(-1)

            # Calculate loss
            loss = self.loss_fn(action_logits.view(-1, action_logits.size(-1)), ground_truth_action_indices)
            probabilities = F.softmax(action_logits, dim=1)

            # Get the predicted class index for each sample
            predicted_classes = torch.argmax(probabilities, dim=-1)
            return action_embeds, loss, predicted_classes
        else:
            
            return action_embeds
        
        
        




        

class CoupledAutoregressiveTransformer(nn.Module):
    def __init__(self, d_model=512, max_traj_len=3, device='cpu', input_dim=512, num_action=133, vis_input_dim=64, dropout_rate=0.1, n_layer=4, n_head=8, **args):
        super(CoupledAutoregressiveTransformer, self).__init__()

        self.d_model = d_model
        self.max_traj_len = max_traj_len
        self.vis_input_dim = vis_input_dim
        self.device = device

        # Learnable positional embeddings
        self.positional_embeddings_a = torch.nn.Embedding(max_traj_len, d_model) 
        self.positional_embeddings_b = torch.nn.Embedding(max_traj_len, d_model)
        
        transformer_configs=GPT2Config()
        transformer_configs.n_positions=self.max_traj_len+2
        transformer_configs.n_ctx=transformer_configs.n_positions
        #### model_dims
        transformer_configs.n_embd=d_model
        transformer_configs.n_layer=n_layer
        transformer_configs.n_head=n_head
        transformer_configs.resid_pdrop=dropout_rate
        transformer_configs.embd_pdrop=dropout_rate
        transformer_configs.attn_pdrop=dropout_rate
        transformer_configs.vocab_size=num_action +100
        self.transformer_configs=transformer_configs

        # Transformer model
        self.transformer_a = GPT2Model(self.transformer_configs).to(device)
        self.transformer_b = GPT2Model(self.transformer_configs).to(device)

        # Projection layer
        self.projection_layer_aa = nn.Sequential(
            nn.Linear(d_model, 2*d_model),  #
            nn.ReLU(),
            nn.Linear(2*d_model, 2*d_model),  # Output shape: n_batch, d_model
            nn.ReLU(),
            nn.Linear(2*d_model, d_model),  #
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.projection_layer_ab = nn.Sequential(
            nn.Linear(d_model, 2*d_model),  #
            nn.ReLU(),
            nn.Linear(2*d_model, 2*d_model),  # Output shape: n_batch, d_model
            nn.ReLU(),
            nn.Linear(2*d_model, d_model),  #
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.projection_layer_ba = nn.Sequential(
            nn.Linear(d_model, 2*d_model),  #
            nn.ReLU(),
            nn.Linear(2*d_model, 2*d_model),  # Output shape: n_batch, d_model
            nn.ReLU(),
            nn.Linear(2*d_model, d_model),  #
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.projection_layer_bb = nn.Sequential(
            nn.Linear(d_model, 2*d_model),  #
            nn.ReLU(),
            nn.Linear(2*d_model, 2*d_model),  # Output shape: n_batch, d_model
            nn.ReLU(),
            nn.Linear(2*d_model, d_model),  #
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # MLP for encoding state
        #self.state_encoder = nn.Sequential(
        #    nn.Conv2d(4, 1, kernel_size=3, stride=2, padding=1),  # Change input channels to 4
        #    nn.ReLU(),
        #    nn.Flatten(),
        #    nn.Linear(32 * 32, d_model),  # Output shape: n_batch, d_model
        #    nn.ReLU(),
        #    nn.Dropout(dropout_rate)
        #)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(input_dim, 2*d_model),  #
            nn.ReLU(),
            nn.Linear(2*d_model, 2*d_model),  # Output shape: n_batch, d_model
            nn.ReLU(),
            nn.Linear(2*d_model, d_model),  #
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Additional layers can be added if necessary
        # Classifier for action prediction
        self.action_classifier  = nn.Sequential(
            nn.Linear(d_model, d_model),  #
            nn.ReLU(),
            nn.Linear(d_model,num_action),  # Output shape: n_batch, d_model
            nn.ReLU(),
        )

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward_run(self, visual_input, ground_truth_action_indices=None, 
                    ground_truth_action_embeds=None, all_action_embeds=None, ground_truth_state_embeds=None,   loss='mse'):
        #n_batch, n_action, 2, w, h, 3
        batch_size = visual_input.shape[0]
        visual_s=visual_input[:, 0,0,...]
        visual_f=visual_input[:,-1,-1,...]
        # Pass visual input through state encoder
        vis_s = self.state_encoder(visual_s)
        vis_f = self.state_encoder(visual_f)
        
        
        ###### a ######
        #get the initial learnble query: 
        idd=torch.LongTensor([0]).to(self.device)
        learnble_query_a = self.positional_embeddings_a(idd).expand(batch_size, -1)
        vis_in_a=torch.cat((vis_s[:,None,:],vis_f[:,None,:], learnble_query_a[:,None,:]),dim=1)
        transformer_output_a = self.transformer_a(inputs_embeds=vis_in_a).last_hidden_state[:,-1,:].clone()
        out_a=[transformer_output_a[:,None,:]]
        
        ###### b ######
        learnble_query_b = self.positional_embeddings_b(idd).expand(batch_size, -1)
        c_b = self.projection_layer_ba(transformer_output_a)+ learnble_query_b
        vis_in_b=torch.cat((vis_s[:,None,:],vis_f[:,None,:], c_b[:,None,:]),dim=1)
        transformer_output_b = self.transformer_b(inputs_embeds=vis_in_b).last_hidden_state[:,-1,:].clone()
        out_b=[transformer_output_b[:,None,:]]
        
        # Iterate through each action
        for i in range(1, self.max_traj_len):
            idd=torch.LongTensor([i]).to(self.device)
            ###### a ######
            learnble_query_a = self.positional_embeddings_a(idd).expand(batch_size, -1)[:,None,:]
            c_a = self.projection_layer_aa(transformer_output_a[:,None,:]) + self.projection_layer_ab(transformer_output_b[:,None,:]) 
            input_embed = c_a + learnble_query_a
            vis_in_a =torch.cat((vis_in_a, input_embed),dim=1)
            transformer_output_a = self.transformer_a(inputs_embeds=vis_in_a).last_hidden_state[:,-1,:].clone()
            out_a.append(transformer_output_a[:,None,:])
            
            ###### b ######
            learnble_query_b = self.positional_embeddings_b(idd).expand(batch_size, -1)[:,None,:]
            c_b = self.projection_layer_ba(transformer_output_a[:,None,:]) + self.projection_layer_bb(transformer_output_b[:,None,:]) 
            input_embed = c_b + learnble_query_b
            vis_in_b =torch.cat((vis_in_b, input_embed),dim=1)
            transformer_output_b = self.transformer_b(inputs_embeds=vis_in_b).last_hidden_state[:,-1,:].clone()
            out_b.append(transformer_output_b[:,None,:])
            
            
            
        action_embeds= torch.cat(out_a,dim=1)
        tstate_embeds= torch.cat(out_b,dim=1)
        # Compute MSE loss between predicted action embeddings and ground truth action embeddings
        if ground_truth_action_embeds!=None and loss=='mse':
            assert all_action_embeds!=None, "Provide all_action_embeds for finding indices under mse mode."
            assert ground_truth_state_embeds!=None, "Provide ground_truth_state_embeds for  mse of states."
            #to do: calculate the loss
            mse_loss_a = F.mse_loss(action_embeds, ground_truth_action_embeds)
            mse_loss_b = F.mse_loss(tstate_embeds, ground_truth_state_embeds)
            mse_loss = mse_loss_a + mse_loss_b
            predicted_indices = torch.argmax(torch.matmul(action_embeds.view(-1, action_embeds.size(-1)), all_action_embeds.T), dim=1).view(action_embeds.size(0), action_embeds.size(1))
            return action_embeds, mse_loss, predicted_indices
        elif loss=='ce' and ground_truth_action_indices!=None:
            # Predict action indices
            action_logits = self.action_classifier(action_embeds)

            # Reshape ground truth for loss calculation
            ground_truth_action_indices = ground_truth_action_indices.view(-1)

            # Calculate loss
            loss = self.loss_fn(action_logits.view(-1, action_logits.size(-1)), ground_truth_action_indices)
            probabilities = F.softmax(action_logits, dim=1)

            # Get the predicted class index for each sample
            predicted_classes = torch.argmax(probabilities, dim=-1)
            return action_embeds, loss, predicted_classes
        else:
            
            return action_embeds
        
    def forward(self, visual_input, ground_truth_action_indices=None,
                ground_truth_action_embeds=None, all_action_embeds=None, ground_truth_state_embeds=None, loss='mse'):
        #n_batch, n_action, 2, w, h, 3
        batch_size = visual_input.shape[0]
        visual_s=visual_input[:, 0,0,...]
        visual_f=visual_input[:,-1,-1,...]
        # Pass visual input through state encoder
        vis_s = self.state_encoder(visual_s)
        vis_f = self.state_encoder(visual_f)
        # get all the learnble queries
        ##### a ######
        learnble_query_a = torch.cat([self.positional_embeddings_a(torch.LongTensor([idd]).to(self.device)).expand(batch_size, -1)[:,None,:]
                          for idd in range(1,self.max_traj_len )], dim=1)
        
        in_act_embeds_a= self.projection_layer_aa(ground_truth_action_embeds[:, :-1, ...]) + self.projection_layer_ab(ground_truth_state_embeds[:, :-1, ...])+ learnble_query_a
        #get the initial learnble query: 
        idd=torch.LongTensor([0]).to(self.device)
        learnble_query = self.positional_embeddings_a(idd).expand(batch_size, -1)
        vis_in_a=torch.cat((vis_s[:,None,:],vis_f[:,None,:], learnble_query[:,None,:], in_act_embeds_a),dim=1)
        action_embeds = self.transformer_a(inputs_embeds=vis_in_a).last_hidden_state[:,-3:,:].clone()
        
        
        ##### b #######
        learnble_query_b = torch.cat([self.positional_embeddings_b(torch.LongTensor([idd]).to(self.device)).expand(batch_size, -1)[:,None,:]
                          for idd in range(1,self.max_traj_len )], dim=1)
        
        in_act_embeds_b= self.projection_layer_ba(ground_truth_action_embeds[:, 1:, ...]) + self.projection_layer_bb(ground_truth_state_embeds[:, :-1, ...])+ learnble_query_b
        #get the initial learnble query: 
        idd=torch.LongTensor([0]).to(self.device)
        learnble_query_b = self.positional_embeddings_b(idd).expand(batch_size, -1)
        in_b_0= learnble_query_b + self.projection_layer_ba(ground_truth_action_embeds[:, 0, ...])
        vis_in_b=torch.cat((vis_s[:,None,:],vis_f[:,None,:], in_b_0[:,None,:], in_act_embeds_b),dim=1)
        tstate_embeds = self.transformer_b(inputs_embeds=vis_in_b).last_hidden_state[:,-3:,:].clone()
        
        
        # Compute MSE loss between predicted action embeddings and ground truth action embeddings
        if ground_truth_action_embeds!=None and loss=='mse':
            assert all_action_embeds!=None, "Provide all_action_embeds for finding indices under mse mode."
            assert ground_truth_state_embeds!=None, "Provide ground_truth_state_embeds for  mse of states."
            #to do: calculate the loss
            mse_loss_a = F.mse_loss(action_embeds, ground_truth_action_embeds)
            mse_loss_b = F.mse_loss(tstate_embeds, ground_truth_state_embeds)
            mse_loss = mse_loss_a + mse_loss_b
            predicted_indices = torch.argmax(torch.matmul(action_embeds.view(-1, action_embeds.size(-1)), all_action_embeds.T), dim=1).view(action_embeds.size(0), action_embeds.size(1))
            return action_embeds, mse_loss, predicted_indices
        elif loss=='ce' and ground_truth_action_indices!=None:
            # Predict action indices
            action_logits = self.action_classifier(action_embeds)

            # Reshape ground truth for loss calculation
            ground_truth_action_indices = ground_truth_action_indices.view(-1)

            # Calculate loss
            loss = self.loss_fn(action_logits.view(-1, action_logits.size(-1)), ground_truth_action_indices)
            probabilities = F.softmax(action_logits, dim=1)

            # Get the predicted class index for each sample
            predicted_classes = torch.argmax(probabilities, dim=-1)
            return action_embeds, loss, predicted_classes
        else:
            
            return action_embeds
            

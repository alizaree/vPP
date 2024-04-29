import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from torch.nn import functional as F

class AutoregressiveTransformer(nn.Module):
    def __init__(self, d_model=512, max_traj_len=3, device='cpu', input_dim=1024, num_action=133, vis_input_dim=64, dropout_rate=0.1, n_layer=4, n_head=8, **args):
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
        self.projection_layer = nn.Linear(d_model, d_model)

        # MLP for encoding state
        self.state_encoder = nn.Sequential(
            nn.Conv2d(4, 1, kernel_size=3, stride=2, padding=1),  # Change input channels to 4
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 32, d_model),  # Output shape: n_batch, d_model
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Additional layers can be added if necessary
        # Classifier for action prediction
        self.action_classifier = nn.Linear(d_model, num_action)

        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, visual_input, ground_truth_action_indices=None, ground_truth_action_embeds=None, loss='mse'):
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
            #to do: calculate the loss
            mse_loss = F.mse_loss(action_embeds, ground_truth_action_embeds)

            return action_embeds, mse_loss
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
            

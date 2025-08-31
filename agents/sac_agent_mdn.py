# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal, Categorical
# import numpy as np

# # --- Critic Network (Unchanged) ---
# class Critic(nn.Module):
#     """
#     Standard Q-Network. Takes a state and action, and returns a Q-value.
#     """
#     def __init__(self, state_dim, action_dim, n_latent_var):
#         super(Critic, self).__init__()
#         self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
#         self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
#         self.layer_3 = nn.Linear(n_latent_var, 1)

#     def forward(self, state, action):
#         x = torch.cat([state, action], 1)
#         x = F.relu(self.layer_1(x))
#         x = F.relu(self.layer_2(x))
#         return self.layer_3(x)

# # --- Actor with Mixture Density Network ---
# class ActorMDN(nn.Module):
#     """
#     An advanced policy network that outputs a mixture of Gaussian distributions.
#     This allows it to represent complex, multi-modal action strategies.
#     """
#     def __init__(self, state_dim, action_dim, n_latent_var, max_action, n_gaussians=5):
#         super(ActorMDN, self).__init__()
#         self.action_dim = action_dim
#         self.n_gaussians = n_gaussians
#         self.max_action = max_action

#         # Common feature extractor layers
#         self.layer_1 = nn.Linear(state_dim, n_latent_var)
#         self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        
#         # Output heads for the Gaussian Mixture Model (GMM) parameters
#         self.pi_head = nn.Linear(n_latent_var, n_gaussians)       # Weights for each Gaussian
#         self.mu_head = nn.Linear(n_latent_var, n_gaussians * action_dim) # Means for each Gaussian
#         self.log_std_head = nn.Linear(n_latent_var, n_gaussians * action_dim) # Log std deviations

#         self.LOG_STD_MAX = 2
#         self.LOG_STD_MIN = -20

#     def forward(self, state):
#         """
#         Processes the state and returns the parameters for the mixture of Gaussians.
#         """
#         x = F.relu(self.layer_1(state))
#         x = F.relu(self.layer_2(x))

#         # Get GMM parameters
#         pi_logits = self.pi_head(x)
#         pi = F.softmax(pi_logits, dim=-1) # Probabilities of choosing each Gaussian
        
#         mu = self.mu_head(x).view(-1, self.n_gaussians, self.action_dim)
        
#         log_std = self.log_std_head(x).view(-1, self.n_gaussians, self.action_dim)
#         log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
#         return pi, mu, log_std

#     def sample(self, state):
#         """
#         Samples an action from the Gaussian mixture and calculates its log probability.
#         """
#         pi, mu, log_std = self.forward(state)
#         std = log_std.exp()

#         # Create the distributions
#         # cat_dist is for choosing *which* Gaussian component to use
#         cat_dist = Categorical(pi)
#         # component_indices will have shape: [batch_size]
#         component_indices = cat_dist.sample()
        
#         # Create a batch of Normal distributions, one for each chosen component
#         batch_indices = torch.arange(state.size(0))
#         mu_selected = mu[batch_indices, component_indices]
#         std_selected = std[batch_indices, component_indices]
#         normal_dist = Normal(mu_selected, std_selected)
        
#         # Sample from the selected Gaussian (with reparameterization trick)
#         x_t = normal_dist.rsample()
#         y_t = torch.tanh(x_t)
#         action = y_t * self.max_action
        
#         # Calculate the log probability for the full Gaussian Mixture Model
#         # This requires combining the probabilities from all components
#         all_component_normals = Normal(mu, std)
#         # Reshape action for broadcasting across all Gaussians: [batch_size, 1, action_dim]
#         log_probs_all_components = all_component_normals.log_prob(x_t.unsqueeze(1))
        
#         # Sum log probs over the action dimension: [batch_size, n_gaussians]
#         log_probs_all_components = log_probs_all_components.sum(dim=-1)

#         # Use the log-sum-exp trick for a numerically stable calculation
#         log_pi = torch.log(pi + 1e-6)
#         gmm_log_prob = torch.logsumexp(log_pi + log_probs_all_components, dim=-1, keepdim=True)
        
#         # Correction for the tanh squashing function
#         gmm_log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6).sum(1, keepdim=True)
        
#         return action, gmm_log_prob

# # --- SAC Agent Implementation ---
# class SACAgent:
#     def __init__(self, env, state_dim, action_dim, n_latent_var, lr, gamma, tau, alpha):
#         self.gamma = gamma
#         self.tau = tau
#         self.alpha = alpha
#         self.max_action = float(env.action_space.high[0])

#         # ! UPDATED: Use the new ActorMDN
#         self.actor = ActorMDN(state_dim, action_dim, n_latent_var, self.max_action).to('cuda')
#         self.critic_1 = Critic(state_dim, action_dim, n_latent_var).to('cuda')
#         self.critic_1_target = Critic(state_dim, action_dim, n_latent_var).to('cuda')
#         self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
#         self.critic_2 = Critic(state_dim, action_dim, n_latent_var).to('cuda')
#         self.critic_2_target = Critic(state_dim, action_dim, n_latent_var).to('cuda')
#         self.critic_2_target.load_state_dict(self.critic_2.state_dict())

#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
#         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
#         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

#     def select_action(self, state):
#         state = torch.FloatTensor(state).unsqueeze(0).to('cuda')
#         action, _ = self.actor.sample(state)
#         return action.detach().cpu().numpy().flatten()

#     def update(self, replay_buffer, batch_size):
#         # The update logic is the same as before. The complexity of the MDN is
#         # cleanly handled inside the actor's .sample() method.
#         state, action, reward, next_state, done = replay_buffer.sample(batch_size)
#         state = torch.FloatTensor(state).to('cuda')
#         action = torch.FloatTensor(action).to('cuda')
#         reward = torch.FloatTensor(reward).unsqueeze(1).to('cuda')
#         next_state = torch.FloatTensor(next_state).to('cuda')
#         done = torch.FloatTensor(done).unsqueeze(1).to('cuda')

#         with torch.no_grad():
#             next_action, next_log_prob = self.actor.sample(next_state)
#             target_q1 = self.critic_1_target(next_state, next_action)
#             target_q2 = self.critic_2_target(next_state, next_action)
#             target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
#             target_q = reward + (1 - done) * self.gamma * target_q

#         current_q1 = self.critic_1(state, action)
#         current_q2 = self.critic_2(state, action)
        
#         critic_1_loss = F.mse_loss(current_q1, target_q)
#         critic_2_loss = F.mse_loss(current_q2, target_q)
#         critic_loss = critic_1_loss + critic_2_loss

#         self.critic_1_optimizer.zero_grad()
#         self.critic_2_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_1_optimizer.step()
#         self.critic_2_optimizer.step()

#         new_action, log_prob = self.actor.sample(state)
#         q1_new = self.critic_1(state, new_action)
#         q2_new = self.critic_2(state, new_action)
#         actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#         for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)












''' CPU BASED CODE (JUPYET)'''
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.distributions import Normal, Categorical
# import numpy as np

# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, n_latent_var):
#         super(Critic, self).__init__()
#         self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
#         self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
#         self.layer_3 = nn.Linear(n_latent_var, 1)

#     def forward(self, state, action):
#         x = torch.cat([state, action], 1)
#         x = F.relu(self.layer_1(x))
#         x = F.relu(self.layer_2(x))
#         return self.layer_3(x)

# class ActorMDN(nn.Module):
#     def __init__(self, state_dim, action_dim, n_latent_var, max_action, n_gaussians=5):
#         super(ActorMDN, self).__init__()
#         self.action_dim = action_dim
#         self.n_gaussians = n_gaussians
#         self.max_action = max_action
#         self.layer_1 = nn.Linear(state_dim, n_latent_var)
#         self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
#         self.pi_head = nn.Linear(n_latent_var, n_gaussians)
#         self.mu_head = nn.Linear(n_latent_var, n_gaussians * action_dim)
#         self.log_std_head = nn.Linear(n_latent_var, n_gaussians * action_dim)
#         self.LOG_STD_MAX = 2
#         self.LOG_STD_MIN = -20

#     def forward(self, state):
#         x = F.relu(self.layer_1(state))
#         x = F.relu(self.layer_2(x))
#         pi_logits = self.pi_head(x)
#         pi = F.softmax(pi_logits, dim=-1)
#         mu = self.mu_head(x).view(-1, self.n_gaussians, self.action_dim)
#         log_std = self.log_std_head(x).view(-1, self.n_gaussians, self.action_dim)
#         log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
#         return pi, mu, log_std

#     def sample(self, state):
#         pi, mu, log_std = self.forward(state)
#         std = log_std.exp()
#         cat_dist = Categorical(pi)
#         component_indices = cat_dist.sample()
#         batch_indices = torch.arange(state.size(0))
#         mu_selected = mu[batch_indices, component_indices]
#         std_selected = std[batch_indices, component_indices]
#         normal_dist = Normal(mu_selected, std_selected)
#         x_t = normal_dist.rsample()
#         y_t = torch.tanh(x_t)
#         action = y_t * self.max_action
        
#         all_component_normals = Normal(mu, std)
#         log_probs_all_components = all_component_normals.log_prob(x_t.unsqueeze(1)).sum(dim=-1)
#         log_pi = torch.log(pi + 1e-6)
#         gmm_log_prob = torch.logsumexp(log_pi + log_probs_all_components, dim=-1, keepdim=True)
        
#         # ! FIX: Perform the subtraction without an in-place operation (x -= y)
#         # Create a new tensor for the final result.
#         squash_correction = torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6).sum(1, keepdim=True)
#         final_log_prob = gmm_log_prob - squash_correction
        
#         return action, final_log_prob

# class SACAgent:
#     def __init__(self, env, state_dim, action_dim, n_latent_var, lr, gamma, tau, alpha, device):
#         self.gamma = gamma
#         self.tau = tau
#         self.alpha = alpha
#         self.device = device
#         self.max_action = float(env.action_space.high[0])

#         self.actor = ActorMDN(state_dim, action_dim, n_latent_var, self.max_action).to(self.device)
#         self.critic_1 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_1_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
#         self.critic_2 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_2_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
#         self.critic_2_target.load_state_dict(self.critic_2.state_dict())

#         self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
#         self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
#         self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

#     def select_action(self, state):
#         state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
#         action, _ = self.actor.sample(state)
#         return action.detach().cpu().numpy().flatten()

#     def update(self, replay_buffer, batch_size):
#         state, action, reward, next_state, done = replay_buffer.sample(batch_size)
#         state = torch.FloatTensor(state).to(self.device)
#         action = torch.FloatTensor(action).to(self.device)
#         reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
#         next_state = torch.FloatTensor(next_state).to(self.device)
#         done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

#         with torch.no_grad():
#             next_action, next_log_prob = self.actor.sample(next_state)
#             target_q1 = self.critic_1_target(next_state, next_action)
#             target_q2 = self.critic_2_target(next_state, next_action)
#             target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
#             target_q = reward + (1 - done) * self.gamma * target_q

#         current_q1 = self.critic_1(state, action)
#         current_q2 = self.critic_2(state, action)
        
#         critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

#         self.critic_1_optimizer.zero_grad()
#         self.critic_2_optimizer.zero_grad()
#         critic_loss.backward()
#         self.critic_1_optimizer.step()
#         self.critic_2_optimizer.step()

#         new_action, log_prob = self.actor.sample(state)
#         q1_new = self.critic_1(state, new_action)
#         q2_new = self.critic_2(state, new_action)
#         actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        
#         self.actor_optimizer.zero_grad()
#         actor_loss.backward()
#         self.actor_optimizer.step()

#         for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
#         for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
#             target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)






import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
import numpy as np

# --- Critic Network Definition ---
# This network learns the Q-value (state-action value).
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(Critic, self).__init__()
        # Layer definitions
        self.layer_1 = nn.Linear(state_dim + action_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        self.layer_3 = nn.Linear(n_latent_var, 1)

    def forward(self, state, action):
        # Concatenate state and action as input
        x = torch.cat([state, action], 1)
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        return self.layer_3(x)

# --- Actor Network with Mixture Density Network (MDN) ---
# This network learns the policy, outputting a mixture of Gaussian distributions
# to represent a complex, potentially multi-modal action distribution.
class ActorMDN(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, max_action, n_gaussians=5):
        super(ActorMDN, self).__init__()
        self.action_dim = action_dim
        self.n_gaussians = n_gaussians
        self.max_action = max_action
        
        # Layer definitions
        self.layer_1 = nn.Linear(state_dim, n_latent_var)
        self.layer_2 = nn.Linear(n_latent_var, n_latent_var)
        
        # MDN Heads
        self.pi_head = nn.Linear(n_latent_var, n_gaussians)      # Mixture weights
        self.mu_head = nn.Linear(n_latent_var, n_gaussians * action_dim)     # Means
        self.log_std_head = nn.Linear(n_latent_var, n_gaussians * action_dim) # Log std deviations

        # Clamp log_std for numerical stability
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

    def forward(self, state):
        # Forward pass through shared layers
        x = F.relu(self.layer_1(state))
        x = F.relu(self.layer_2(x))
        
        # Get MDN parameters
        pi_logits = self.pi_head(x)
        pi = F.softmax(pi_logits, dim=-1)
        mu = self.mu_head(x).view(-1, self.n_gaussians, self.action_dim)
        log_std = self.log_std_head(x).view(-1, self.n_gaussians, self.action_dim)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        
        return pi, mu, log_std

    def sample(self, state):
        pi, mu, log_std = self.forward(state)
        std = log_std.exp()

        # Create a categorical distribution to sample a Gaussian component
        cat_dist = Categorical(pi)
        component_indices = cat_dist.sample()

        # Select the mu and std for the chosen component
        batch_indices = torch.arange(state.size(0))
        mu_selected = mu[batch_indices, component_indices]
        std_selected = std[batch_indices, component_indices]

        # Sample from the selected Gaussian
        normal_dist = Normal(mu_selected, std_selected)
        x_t = normal_dist.rsample()  # rsample for reparameterization trick

        # Apply tanh squashing and scale to action range
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # --- Calculate Log Probability of the Action ---
        # This is complex for a GMM but required for SAC
        all_component_normals = Normal(mu, std)
        log_probs_all_components = all_component_normals.log_prob(x_t.unsqueeze(1)).sum(dim=-1)
        log_pi = torch.log(pi + 1e-6)
        gmm_log_prob = torch.logsumexp(log_pi + log_probs_all_components, dim=-1, keepdim=True)
        
        # Correction for the tanh squashing
        squash_correction = torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6).sum(1, keepdim=True)
        final_log_prob = gmm_log_prob - squash_correction
        
        return action, final_log_prob

# --- Soft Actor-Critic (SAC) Agent ---
class SACAgent:
    def __init__(self, env, state_dim, action_dim, n_latent_var, lr, gamma, tau, alpha, device):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.device = device # Store the device (e.g., 'cuda' or 'cpu')
        self.max_action = float(env.action_space.high[0])

        # --- GPU ACCELERATION STEP 1: Move all networks to the designated device ---
        self.actor = ActorMDN(state_dim, action_dim, n_latent_var, self.max_action).to(self.device)
        
        self.critic_1 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_1_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_2_target = Critic(state_dim, action_dim, n_latent_var).to(self.device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        # Initialize optimizers for networks that are now on the GPU
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=lr)

    def select_action(self, state):
        with torch.no_grad():
            # --- GPU ACCELERATION STEP 2: Move input data to the GPU for inference ---
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor.sample(state_tensor)
            
            # --- GPU ACCELERATION STEP 3: Move action back to CPU to interact with the environment ---
            return action.detach().cpu().numpy().flatten()

    def update(self, replay_buffer, batch_size):
        # Sample a batch from the replay buffer (data is on CPU)
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        # --- GPU ACCELERATION STEP 4: Move the entire batch to the GPU for training ---
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)

        # --- All subsequent calculations now happen on the GPU ---

        # Calculate target Q-value
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q

        # Get current Q estimates
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        
        # Calculate critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Optimize the critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()

        # Calculate actor loss
        new_action, log_prob = self.actor.sample(state)
        q1_new = self.critic_1(state, new_action)
        q2_new = self.critic_2(state, new_action)
        actor_loss = (self.alpha * log_prob - torch.min(q1_new, q2_new)).mean()
        
        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for target_param, param in zip(self.critic_1_target.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic_2_target.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

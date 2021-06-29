import numpy as np
from numpy.core.numeric import Inf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

#Agent model
class Agent_1V():

    def __init__(self, sensory_buffer_size, episodic_buffer_size, reward_location_buffer_size, len_1D_maze, theta_encoding_len, params):
        self.sensory_buffer_size = sensory_buffer_size
        self.sensory_buffer = []
        self.episodic_buffer_size = episodic_buffer_size
        self.episodic_buffer = []
        self.reward_location_buffer_size = reward_location_buffer_size
        self.reward_location_buffer = []
        self.len_1D_maze = len_1D_maze
        self.theta_encoding_len = theta_encoding_len
        self.proximals = None
        self.params = params

        self.state_values = np.zeros((self.len_1D_maze))
        self.plan_values = np.zeros((self.len_1D_maze))
        self.stored_values = np.zeros((self.len_1D_maze))
        self.goal_values = np.zeros((self.len_1D_maze))
        self.reward_model = np.zeros((self.len_1D_maze))
        self.confidence_matrix = np.zeros((self.len_1D_maze, self.len_1D_maze))


    # values that needs to be restored to initial conditions after each episode
    def reset_each_episode(self, position):
        self.sensory_buffer = [position] * self.sensory_buffer_size
        self.goal_values = np.zeros((self.len_1D_maze))
        self.plan_values = np.zeros((self.len_1D_maze))

    # insert a new observation into the sensory buffer and remove the old one if it exceeds the maximal capacity
    def update_sensory_buffer(self, new_observation):
        self.sensory_buffer.append(new_observation)
        if len(self.sensory_buffer) > self.sensory_buffer_size:
            self.sensory_buffer.pop(0)

    # insert a new observation into the reward location buffer and remove the old one if it exceeds the maximal capacity
    def update_reward_location_buffer(self, reward_location):
        self.reward_location_buffer.append(reward_location)
        if len(self.reward_location_buffer) > self.reward_location_buffer_size:
            self.reward_location_buffer.pop(0)

    # insert a new episode into the episodic memory and remove the old one if it exceeds the maximal capacity
    # (each episode is composed by both the input sequence and the labels extracted from the same transition sequence experienced)
    def update_episodic_buffer(self, lesion=None):
        experiences = [self.sensory_buffer[(-self.theta_encoding_len-1):-1], self.sensory_buffer[-self.theta_encoding_len:]].copy()     
        if lesion:
            experiences = [[0]*self.theta_encoding_len, [0]*self.theta_encoding_len]
        self.episodic_buffer.append(experiences)
        overload = len(self.episodic_buffer) - self.episodic_buffer_size
        if overload > 0:
            self.episodic_buffer = self.episodic_buffer[overload:]

    # one-shot learning of the state-reward representaiton
    def update_reward_model(self, observation, reward):
        self.reward_model[observation] = reward

    # randomly sample a batch of n episodes form the episodic buffer
    def sample_from_episodic_buffer(self, size_batch):
        random_idxs = np.random.randint(0,len(self.episodic_buffer), size_batch)
        sample_inputs = []
        sample_targets = []
        for i in random_idxs:
            sample_inputs.append(self.episodic_buffer[i][0].copy())
            sample_targets.append(self.episodic_buffer[i][1].copy())
        return [sample_inputs, sample_targets]

    # from the episodic buffer extract the episodes that contain the reward location
    def seed_from_episodic_buffer(self, max_size_batch):
        indices = [i for i, s in enumerate(self.episodic_buffer) if s[1][-1] in list(np.where(self.reward_model == 1)[0])]
        extracted = list(self.episodic_buffer[x].copy() for x in indices)

        #randomly sample from them a batch of n episodes
        len_extr = len(extracted)
        sample_inputs = []
        sample_targets = []
        if len_extr <= 0:
            return 0
        random_idxs = np.random.randint(0, len_extr, max_size_batch)
        for i in random_idxs:
            sample_inputs.append(extracted[i][0])
            sample_targets.append(extracted[i][1])
        return [sample_inputs, sample_targets]

    # compute the contribution of SWR sequences (stored values) to state values accordingly to the content of a given set of SWR sequences
    def SWR_update_stored_values(self, SWR_batch):
        stored_add = np.zeros((len(self.stored_values)))
        for count, seq in enumerate(SWR_batch):
            credit = 1
            for idx, pos in enumerate(seq):
                credit = credit * 0.9
                stored_add[pos] = credit
            stored_add = stored_add / len(SWR_batch)
            self.stored_values = self.stored_values + 0.03 * stored_add     #0.03 is the factor of update of stored vaues
        
    # compute the contribution of Theta sequences (plan values) to state values accordingly to the content of a given set of Theta sequences evalued in respect to a goal state
    def theta_evaluation_goal(self, theta_batch, goal):
        plan_add = np.zeros((len(self.plan_values)))
        for count, seq in enumerate(theta_batch):
            credit = 1
            if goal in seq[-4:] or goal-1 in seq[-4:]:
                for idx, pos in enumerate(seq):
                    credit = credit * 0.9
                    plan_add[pos] += credit
                    if pos == goal:
                        break
                self.plan_values = self.plan_values + plan_add

    # integrate different contributions to state values (used to produce T2 and S2 lesions)
    def integrate_state_values(self, condition="Int"):

        stored_norm = self.stored_values
        plan_norm = self.plan_values

        if condition == "Int":
            self.state_values = stored_norm + plan_norm
        elif condition == "Stored":
            self.state_values = stored_norm.copy()
        elif condition == "Planned":
            self.state_values = plan_norm.copy()
            
    # sample from a certain probability distribution described by a given array
    # the maximal value is preserved and other values are adjusted to have the total array sum equal to 1
    def softmax_from_array(self, array, n_samples=1):
        p = array.astype('float32')
        
        if any(p > 1) and len(p) > 1:
            p = (p - min(p)) / (max(p)-min(p))  #normalization if values exceeds 1
        u = p+0.0001    #to avoid division by 0
        u[u>1] = 1
        other = u.copy()
        other[np.argmax(u)] = 0
        other = (other / other.sum()) * (1-max(u))
        other[np.argmax(u)] = max(u)      
        p = other
        if p.sum() > 0.01:  #always true in this case
            p = p / p.sum()
            p = np.nan_to_num(p, nan=0)
            idx = np.arange(len(p))
            samples = np.random.choice( idx, n_samples, p=p)
        else:
            idx = np.arange(len(p))
            samples = np.random.choice(idx, n_samples)
        return samples

    # define goal values in the base of the reward model (here it could be added other methods of setting goals, like novelty preference / curiosity)
    def compute_goal_values(self):
        rew_pos = np.argmax(self.reward_model)
        self.goal_values[rew_pos] += 1

    # compute the successor matrix (see original document for explanation)
    def mean_confidence_space(self, theta_softmax_pred, theta_pred, observation):
        void_space = np.zeros((self.len_1D_maze))
        for [c_seq, possible_seq] in enumerate(theta_pred):
            void_count = np.zeros((self.len_1D_maze))
            void_add = np.zeros((self.len_1D_maze))
            pos_dependent_confidence = 1
            gamma_discount = 1
            for [c_pos, pos] in enumerate(possible_seq):
                pos_dependent_confidence = pos_dependent_confidence * theta_softmax_pred[c_seq][c_pos][0][pos] * gamma_discount
                void_add[pos] += pos_dependent_confidence
                void_count[pos] += 1
            gamma_discount = gamma_discount * 0.986
            void_count[void_count == 0] = 1  
            void_add = void_add / void_count
            void_space += void_add
        conf_space = void_space / len(theta_pred)
        self.confidence_matrix[observation, :] = (9*self.confidence_matrix[observation, :] + conf_space) /10
        return self.confidence_matrix


    # BEHAVIOR ---------------------------------------------------------------------------------------------------------------------------------------------------------------

    # choose the next state randomly form proximals
    def behavior_random_policy(self):
        return np.random.choice(self.proximals)

    # choose the next state from a given path to follow
    def behavior_fixed_policy(self, current, path):
        here = path.index(current)
        next_state = path[here+1]
        return next_state

    # choose the next state based on integrated state values, only SWR contribution or only Theta contribution
    def behavior_policy_values(self, values = "state_values"):
        proximal_states_directional = self.proximals.copy()
        if len(self.proximals) > 1:
            proximal_states_directional = np.delete(self.proximals, np.where(self.proximals==self.sensory_buffer[-2])[0])
        if values == "state_values":
            proximal_values = self.state_values[proximal_states_directional]
            idx = self.softmax_from_array(proximal_values)[0]
        elif values == "stored":
            proximal_values = self.stored_values[proximal_states_directional]
            idx = self.softmax_from_array(proximal_values)[0]
        elif values == "on_the_fly":
            proximal_values = self.plan_values[proximal_states_directional]
            idx = np.argmax(proximal_values)
        return proximal_states_directional[idx]

    # choose the next state randomly from the proximals states with the exception of returning to the same state from which the agent is coming from
    def behavior_pseudo_random_policy(self):
        proximal_states_directional = self.proximals.copy()
        if len(self.proximals) > 1:
            proximal_states_directional = np.delete(self.proximals, np.where(self.proximals==self.sensory_buffer[-2])[0])
        return np.random.choice(proximal_states_directional)



    #SEQUENCES FUNCTIONS ------------------------------------------------------------------------------------------------------------------------------------------------------

    # depending on the network state
    # Theta state: takes sensory sequence as input, pass it through the model and produce an output predicion (first part of the theta sequence) - learing rate is minimal and doesn't produce almost any learning in the networks
    # SWR state: takes a batch of episodic sequences as input, pass it through the model and produce an output prediciton (first part of SWR sequence) - learning rate is optimal to produce learning in the networks
    def Encoding(self, start_params, len_1D_maze, netF, netB, optimizerF, optimizerB, one_hot_encoder, device, loss_fn, SWR=False, ep=Inf):

        if SWR: #If SWR state -> learning rate is optimal
            b_size = start_params['SWR_learning_episodic_batch_size']
            if ep < start_params["latent_ep"]:
                b_size = start_params['SWR_learning_episodic_batch_size'] * 5     # -> 120
            swr_sample = self.sample_from_episodic_buffer(b_size)
            if any(self.reward_model == 1):
                selected = self.seed_from_episodic_buffer(1)
                if selected != 0:
                    swr_sample[0][0] = selected[0][0].copy()
                    swr_sample[1][0] = selected[1][0].copy()
                
                for g in optimizerF.param_groups:
                    g['lr'] = start_params['learning_rate']
            
                for g in optimizerB.param_groups:
                    g['lr'] = start_params['learning_rate']
        else:   # If Theta State -> learning rate is minimal
            for g in optimizerF.param_groups:
                g['lr'] = start_params['learning_rate'] / 100
            
            for g in optimizerB.param_groups:
                g['lr'] = start_params['learning_rate'] / 100

        #during latent learning episodes the learning rate is larger
        if ep < start_params["latent_ep"]:
            for g in optimizerF.param_groups:
                g['lr'] =  start_params['learning_rate'] * 100

            for g in optimizerB.param_groups:
                g['lr'] = start_params['learning_rate'] * 100


        # FORWARD NETWORK
        # Prepare Input forward
        episodic_sample = self.sample_from_episodic_buffer(start_params['learning_episodic_batch_size'])
        previous_theta_batch = episodic_sample[0]
        previous_theta_batch[0] = self.sensory_buffer[(-start_params['theta_encoding_len']-1):-1]
        if SWR:
            previous_theta_batch = swr_sample[0]
        len_batch = len(previous_theta_batch)
        #convert in one-hot tensor
        previous_theta_batch_tensor = torch.zeros([len_batch, start_params['theta_encoding_len'], len_1D_maze])
        for [i, sample] in enumerate(previous_theta_batch):
            onehot = one_hot_encoder(sample)
            previous_theta_batch_tensor[i] = torch.tensor(onehot).float()
        # Prepare Labels forward
        theta_batch_labels = episodic_sample[1]
        theta_batch_labels[0] = self.sensory_buffer[-start_params['theta_encoding_len']:]
        if SWR:
            theta_batch_labels = swr_sample[1]
        # convert in one-hot tensor
        theta_batch_tensor_labels = torch.zeros([len_batch, start_params['theta_encoding_len'], len_1D_maze])
        for [i, sample] in enumerate(theta_batch_labels):
            onehot = one_hot_encoder(sample)
            theta_batch_tensor_labels[i] = torch.tensor(onehot).float()
        # set training mode
        netF.train()
        net_input = previous_theta_batch_tensor.to(device)
        labels = theta_batch_tensor_labels.to(device)
        optimizerF.zero_grad()
        # Forward pass
        net_outF, rnn_stateF = netF(net_input)
        # evaluate forward loss
        labels = labels.argmax(dim=-1)
        net_outF = net_outF.permute([0,2,1])
        loss_F = loss_fn(net_outF, labels)


        # BACKWARD NETWORK (reversed input and labels, in reverse order)
        # Prepare Labels reversed
        episodic_sample = self.sample_from_episodic_buffer(start_params['learning_episodic_batch_size'])
        previous_theta_batch = episodic_sample[0]
        previous_theta_batch[0] = self.sensory_buffer[(-start_params['theta_encoding_len']-1):-1]
        if SWR:
            previous_theta_batch = swr_sample[0]
        len_batch = len(previous_theta_batch)
        #convert in one-hot tensor
        theta_batch_tensor_labels = torch.zeros([len_batch, start_params['theta_encoding_len'], len_1D_maze])
        for [i, sample] in enumerate(previous_theta_batch):
            sample = sample.copy()
            sample.reverse()
            onehot = one_hot_encoder(sample)
            theta_batch_tensor_labels[i] = torch.tensor(onehot).float()
        # Prepare Input reversed
        theta_batch_labels = episodic_sample[1] 
        theta_batch_labels[0] = self.sensory_buffer[-start_params['theta_encoding_len']:]
        if SWR:
            theta_batch_labels = swr_sample[1]
        # convert in one-hot tensor
        previous_theta_batch_tensor = torch.zeros([len_batch, start_params['theta_encoding_len'], len_1D_maze])
        for [i, sample] in enumerate(theta_batch_labels):
            sample = sample.copy()
            sample.reverse()
            onehot = one_hot_encoder(sample)
            previous_theta_batch_tensor[i] = torch.tensor(onehot).float()
        # set training mode
        netB.train()
        net_input = previous_theta_batch_tensor.to(device)
        labels = theta_batch_tensor_labels.to(device)
        optimizerB.zero_grad()
        # Backward pass
        net_outB, rnn_stateB = netB(net_input)
        # Evaluate loss
        labels = labels.argmax(dim=-1)
        net_outB = net_outB.permute([0,2,1])
        loss_B = loss_fn(net_outB, labels)

        # COMMON TO BOTH NETWORKS
        loss = loss_F + loss_B
        # Backward pass
        loss.backward()
        # Update weights
        optimizerF.step()
        optimizerB.step()
        # Save batch loss
        batch_loss = loss.data.cpu().numpy()

        #Get the softmax sampling
        if not SWR:
            net_out = net_outF #forward in the Theta state
        else:
            net_out = net_outB  #backward in the SWR state
        net_out = net_out.permute([0,2,1])
        theta_enc = []
        theta_softmax_enc = []
        for pos in range(start_params['theta_encoding_len']):
          sotmax_dist = Categorical(nn.functional.softmax(net_out[0,pos,:], dim=0))
          next_pos_encoded = sotmax_dist.sample().item()
          theta_enc.append(next_pos_encoded)
          theta_softmax_enc.append(sotmax_dist.probs.detach().cpu().numpy())

        return theta_enc, batch_loss, rnn_stateF, rnn_stateB, theta_softmax_enc



    # Takes as input the first part of Theta or SWR sequence (depending on the network state) and generate the successive states with the knowledge contained in the forward or backward network (depending on the network state)
    def Prediction(self, start_params, netF, netB, device, c_transform, seed, net_state=None, SWR=False):

        theta_predict = []
        theta_softmax_pred = []
        next_pos = [seed[-1]]
        #lenght of generation
        len_prediction_seq = start_params['theta_prediction_len']
        if SWR:
            len_prediction_seq = start_params['SWR_prediction_len']

        #generative loop
        for firing_cell in range(len_prediction_seq):
            with torch.no_grad():
                # Prepare input
                net_input = c_transform(next_pos)
                net_input = net_input.to(device)
                net_input = net_input.unsqueeze(0)
                if net_state == None:           
                    if not SWR:
                        net_out, net_state = netF(net_input)    #If theta state, use forward predicions 
                    else:
                        net_out, net_state = netB(net_input)    #If SWR state, use backward predicions 
                else: #if the content LSTM state is given provided to the network           
                    if not SWR:
                        net_out, net_state = netF(net_input, net_state) #If theta state, use forward predicions 
                    else:
                        net_out, net_state = netB(net_input, net_state) #If SWR state, use backward predicions 
                #Get the softmax sampling
                net_out = torch.squeeze(net_out, 0)
                top = torch.topk(net_out, 10)[0][0]
                diff = top[0]-top[1]
                net_out[net_out < net_out.min()+diff] = (net_out.min()+diff)
                sotmax_dist = Categorical(nn.functional.softmax(net_out, dim=1))
                next_pos_encoded = sotmax_dist.sample().item()
                next_pos = [next_pos_encoded]
                theta_predict.append(next_pos_encoded)
                theta_softmax_pred.append(sotmax_dist.probs.detach().cpu().numpy())
        return theta_predict, theta_softmax_pred


    # Integrate the bottom-up information with the tow-down prediction
    # The integration is the mean of the probability distributions
    def Integration(self, theta_enc, theta_softmax_enc, one_hot_encoder, SWR=False):
      bottom_up = self.sensory_buffer[-len(theta_enc):]
      if SWR:
          bottom_up = self.sensory_buffer[(-len(theta_enc)-1):-1].copy()
          bottom_up.reverse()
      top_down = theta_softmax_enc
      theta_int = []
      theta_softmax_int = []
      for [i, top_down_dist] in enumerate(top_down):
        bottom_up_dist = one_hot_encoder([bottom_up[i]])[0]
        mixed_dist = (top_down_dist + bottom_up_dist) / 2
        theta_softmax_int.append([mixed_dist])
        #sampling
        p = mixed_dist
        p = p / p.sum()
        p = np.nan_to_num(p, nan=0)
        idx = np.arange(len(p))
        sample = np.random.choice( idx, 1, p=p)[0]
        theta_int.append(sample)
      return theta_int, theta_softmax_int
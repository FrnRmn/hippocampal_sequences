#----------------------IMPORTS-----------------------------------
#general
import numpy as np
import matplotlib.pyplot as plt
import pickle

#torch
import torch
import torch.nn as nn
from torchvision import transforms

#personal
from environments import corridor
from utils import OneHotEncoder, ToTensor
from agents_dual import Agent_1V
from lstm_model import Network

with open('C1_start_params.pickle', 'rb') as handle:
    start_params = pickle.load(handle)

# SIMULATION FACTORS ---------------------------------------------------------
reward_factor = "absent" # present, absent
policy_factor = "randomwalk"  # randomwalk, directional
# ----------------------------------------------------------------------------

maze_matrix = np.zeros((1,30)) + 1
maze_matrix[0,-1] = 2

maze_code = {'wall' : 0,
'floor' : 1,
'possible_rewards' : 2,
'multidim_end' : 3,
'multidim_start' : 4}

reward_pos = 0
counter_task = 0

#----------------------CLASS OBJECTS INITIALIZATION-----------------------------------
env = corridor(maze_matrix, maze_code)
len_1D_maze = len(env.vector)

c_transform = transforms.Compose([OneHotEncoder(len_1D_maze),ToTensor()])
one_hot_encoder = OneHotEncoder(len_1D_maze)

agent = Agent_1V(start_params['sensory_buffer_size'], start_params['episodic_buffer_size'],  start_params['reward_location_buffer_size'], len_1D_maze, start_params['theta_encoding_len'], start_params)

netF = Network(len_1D_maze, start_params['hidden_units'], start_params['layers_num'], start_params['dropout_prob'])
netB = Network(len_1D_maze, start_params['hidden_units'], start_params['layers_num'], start_params['dropout_prob'])
optimizerF = torch.optim.RMSprop(netF.parameters(), lr=start_params['learning_rate'], weight_decay=start_params['weight_decay'])
optimizerB = torch.optim.RMSprop(netB.parameters(), lr=start_params['learning_rate'], weight_decay=start_params['weight_decay'])
loss_fn = nn.CrossEntropyLoss()
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")    #with cuda the code might require slight changes
device = torch.device("cpu")
print(f"Training device: {device}")
netF.to(device)
netB.to(device)

#----------------------LOOP-----------------------------------
rnn_state = 0
reward_history_tot = []
loss_tot = []
behavior_tot = []
theta_batch_tot = []
theta_softmax_tot = []
SWR_tot = []
SWR_softmax_tot = []
stored_values_tot = []
on_the_fly_values_tot = []
confidence_matrix_tot = []
#----------------------START LOOP EPISODES-----------------------------------
for ep in range(start_params["n_ep"]):
    rnn_state = 0 
    
    #Set agent to starting position
    if policy_factor == "directional":
      env.set_agent_position(start_params["agent_starting_position"])
    elif policy_factor == "randomwalk":
      random_pos = int(np.random.choice(np.arange(len(env.vector)), 1)[0])
      env.set_agent_position(random_pos)

    #Reward is placed according to the task
    if reward_factor == "present":
      reward_pos = len_1D_maze-1
    elif reward_factor == "absent":
      reward_pos = -1
    reward_history_tot.append(reward_pos)

    #Reset agent for initial episode
    agent.reset_each_episode(env.agent)

    network_state = "Theta"
    loss_ep = []
    behavior_ep = []
    theta_batch_ep = []
    theta_softmax_ep = []
    SWR_ep = []
    SWR_softmax_ep = []
    stored_values_ep = []
    on_the_fly_values_ep = []
    #----------------------START LOOP BEHAVIORAL STEPS-----------------------------------
    step = 0
    while True:
        behavior_ep.append(env.agent)

        #Observations from the world (directly agent position and proximal states)
        observation = env.agent
        agent.proximals = env.agent_available_proximal_states()
        reward = env.check_reward(reward_pos)


        if agent.reward_model[observation] != reward:
            agent.update_reward_model(observation, reward)
        if reward == 1:
            network_state = "SWR"
            agent.update_reward_location_buffer(observation)
            agent.stored_values[observation] = 1
            

        #Update sensory buffer
        agent.update_sensory_buffer(observation)

        #Update episodic buffer
        agent.update_episodic_buffer()

        #----------------------START LOOP SEQUENCES-----------------------------------
        theta_enc = [0] * start_params['theta_compression_factor']
        theta_softmax_enc = [0] * start_params['theta_compression_factor']
        theta_int = [0] * start_params['theta_compression_factor']
        theta_softmax_int = [0] * start_params['theta_compression_factor']
        theta_pred = [0] * start_params['theta_compression_factor']
        theta_softmax_pred = [0] * start_params['theta_compression_factor']
        theta_batch = [0] * start_params['theta_compression_factor']
        theta_batch_softmax = [0] * start_params['theta_compression_factor']
        SWR_batch = []
        SWR_softmax_batch = []
        seq_loss = []
        
        
        if network_state == "Theta":
            for pred in range(start_params['theta_compression_factor']):
                
                # FIRST THETA HALF - ENCODING
                ep_sample = agent.sample_from_episodic_buffer(start_params['learning_episodic_batch_size'])
                [theta_enc[pred], batch_loss, rnn_stateF, rnn_stateB, theta_softmax_enc[pred]] = agent.Encoding(start_params, len_1D_maze,  netF, netB, optimizerF, optimizerB, one_hot_encoder, device, loss_fn)
                rnn_state = (rnn_stateF[0][:,0,:].unsqueeze(1), rnn_stateF[1][:,0,:].unsqueeze(1))

                # FIRST THETA PART - Extra - INTEGRATION
                [theta_int[pred], theta_softmax_int[pred]]  = agent.Integration(theta_enc[pred], theta_softmax_enc[pred], one_hot_encoder)

                # SECOND THETA PART - FUTURE PREDICTION
                [theta_pred[pred], theta_softmax_pred[pred]] = agent.Prediction(start_params, netF, netB, device, c_transform, theta_int[pred], rnn_state) #changed theta_enc[pred] into theta_int[pred]

                # COMPLETE THETA BATCH (Multiple Thetas with Encoding + Prediction parts)
                theta_batch[pred] = theta_int[pred] + theta_pred[pred]
                theta_batch_softmax[pred] = theta_softmax_int[pred] + theta_softmax_pred[pred]

                seq_loss.append(batch_loss)

            theta_batch_ep.append(theta_batch)
            theta_softmax_ep.append(theta_batch_softmax)

            agent.mean_confidence_space(theta_softmax_pred, theta_pred, observation)
            
            if len(agent.reward_location_buffer)>0:
                # PLANNING
                agent.compute_goal_values()
                goal_state = np.argmax(agent.goal_values)
                agent.theta_evaluation_goal(theta_pred, theta_softmax_pred, goal_state)


            SWR_ep = []
            SWR_softmax_ep = []

                
        elif network_state == "SWR":

            for pred in range(start_params["SWR_compression_factor"]):
                # FIRST SWR HALF - ENCODING
                [SWR_enc, batch_loss_SWR, rnn_stateF, rnn_stateB, SWR_softmax_enc] = agent.Encoding(start_params, len_1D_maze,  netF, netB, optimizerF, optimizerB, one_hot_encoder, device, loss_fn, SWR=True)
                rnn_stateB = (rnn_stateB[0][:,0,:].unsqueeze(1), rnn_stateB[1][:,0,:].unsqueeze(1))

                
                # FIRST SWR PART - Extra - INTEGRATION
                [SWR_int, SWR_softmax_int]  = agent.Integration(SWR_enc, SWR_softmax_enc, one_hot_encoder, SWR=True)

                # "reverse predicting" with SWR
                [SWR_pred, SWR_softmax_pred] = agent.Prediction(start_params, netF, netB, device, c_transform, SWR_int, rnn_stateB, SWR=True)
                
                # COMPLETE THETA BATCH (Multiple Thetas with Encoding + Prediction parts)
                SWR_seq = SWR_int + SWR_pred
                SWR_seq_softmax = SWR_softmax_int + SWR_softmax_pred

                
                SWR_batch.append(SWR_seq)
                SWR_softmax_batch.append(SWR_seq_softmax)

            
            # add values update of SWR sequences to inform action (model free)
            if any(agent.reward_model == 1):
                agent.SWR_update_stored_values(SWR_batch, SWR_softmax_batch)

            SWR_ep.append(SWR_batch)
            SWR_softmax_ep.append(SWR_softmax_batch)          

        #----------------------FINISH LOOP SEQUENCES-----------------------------------


        #Behavior
        if policy_factor == "directional":
          next_state = observation + 1
        elif policy_factor == "randomwalk":
          proximal_states = agent.proximals.copy()
          next_state = np.random.choice(proximal_states)

        #Environment Step
        env.set_agent_position(next_state)

        #End Conditions
        if env.vector[observation] == 2 and policy_factor == "directional":
            break
        if step >= 29 and policy_factor == "randomwalk":
            break

        step += 1
        loss_ep.append(np.mean(seq_loss))
        stored_values_ep.append(agent.stored_values)
        on_the_fly_values_ep.append(agent.plan_values)
        #----------------------FINISH LOOP STEPS-----------------------------------


    #SELECT MEASURES TO STORE
    loss_tot.append(np.mean(loss_ep))
    behavior_tot.append(behavior_ep)
    theta_batch_tot.append(theta_batch_ep)
    confidence_matrix_tot.append(agent.confidence_matrix)
    #theta_softmax_tot.append(theta_softmax_ep)
    #SWR_tot.append(SWR_ep)
    #SWR_softmax_tot.append(SWR_softmax_ep)
    #stored_values_tot.append(stored_values_ep)
    #on_the_fly_values_tot.append(on_the_fly_values_ep)
    

    #print(step)
    print(f"Epoch {ep + 1} loss: {np.mean(loss_ep)}")
    #----------------------FINISH LOOP EPISODES-----------------------------------


 #---------------------- SAVE MEASURES -----------------------------------
import os

os.chdir('./C1_produced_data')

id_label = "Corridor_" + str(policy_factor) + "_" + str(reward_factor)

with open("behavior_tot"+id_label+".txt", "wb") as fp:   #Bahavior
  pickle.dump(behavior_tot, fp)

with open("theta_batch_tot"+id_label+".txt", "wb") as fp:   #Theta Batch Seq
  pickle.dump(theta_batch_tot, fp)

with open("confidence_matrix_tot"+id_label+".txt", "wb") as fp:   #Theta Batch Seq
  pickle.dump(confidence_matrix_tot, fp)

with open("loss_tot"+id_label+".txt", "wb") as fp:   #Loss
  pickle.dump(loss_tot, fp)

os.chdir("..")


#Quick plot at the end
plt.plot(loss_tot)
plt.title('LSTM Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.show()
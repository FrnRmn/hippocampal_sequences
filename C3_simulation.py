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
from environments import Arm8_maze
from utils import OneHotEncoder, ToTensor
from agents_dual import Agent_1V
from lstm_model import Network

with open('C3_start_params.pickle', 'rb') as handle:
    start_params = pickle.load(handle)

params_bkp = start_params.copy()

maze_matrix = np.load('8arm_map.npy')

maze_code = {'wall' : 0,
'floor' : 1,
'possible_rewards' : 2,
'multidim_end' : 3,
'multidim_start' : 4}


# SIMULATION FACTORS ---------------------------------------------------------
agent_type1 = "Int"      #Int, No_SWR, No_Theta   =    #Normal, S1, T1
agent_type2 = "Int"      #Int, Stored, Planned    =    #Normal, T2, S2
# ----------------------------------------------------------------------------


#----------------------META-LOOP-----------------------------------

behavior_loop = []
reward_history_loop = []
loss_loop = []
meta_loops = start_params["n_agents"]

for ml in range(meta_loops):

    reward_pos = 0
    counter_task = 0
    #----------------------CLASS OBJECTS INITIALIZATION-----------------------------------
    env = Arm8_maze(maze_matrix, maze_code)
    len_1D_maze = len(env.vector)

    c_transform = transforms.Compose([OneHotEncoder(len_1D_maze),ToTensor()])
    one_hot_encoder = OneHotEncoder(len_1D_maze)

    agent = Agent_1V(start_params['sensory_buffer_size'], start_params['episodic_buffer_size'],  start_params['reward_location_buffer_size'], len_1D_maze, start_params['theta_encoding_len'], start_params)

    netF = Network(len_1D_maze, start_params['hidden_units'], start_params['layers_num'], start_params['dropout_prob'])
    netB = Network(len_1D_maze, start_params['hidden_units'], start_params['layers_num'], start_params['dropout_prob'])
    optimizerF = torch.optim.RMSprop(netF.parameters(), lr=start_params['learning_rate'], weight_decay=start_params['weight_decay'])
    optimizerB = torch.optim.RMSprop(netB.parameters(), lr=start_params['learning_rate'], weight_decay=start_params['weight_decay'])
    loss_fn = nn.CrossEntropyLoss()
    #device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
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
    #----------------------START LOOP EPISODES-----------------------------------
    for ep in range(start_params['n_ep']):
        
        if ep < start_params["latent_ep"]:
            start_params["theta_prediction_len"] = start_params["theta_prediction_len_latent"]
            start_params["theta_compression_factor"] = start_params["theta_compression_factor_latent"]
            start_params["SWR_prediction_len"] = start_params["SWR_prediction_len_latent"]
            start_params["SWR_compression_factor"] = start_params["SWR_compression_factor_latent"]
        else:
            start_params["theta_prediction_len"] = params_bkp["theta_prediction_len"]
            start_params["theta_compression_factor"] = params_bkp["theta_compression_factor"]
            start_params["SWR_prediction_len"] = params_bkp["SWR_prediction_len"]
            start_params["SWR_compression_factor"] = params_bkp["SWR_compression_factor"]

        rnn_state = 0

        #Set agent to starting position
        env.set_agent_position(start_params['agent_starting_position'])

        #Reward is placed according to the task
        if ep < start_params["latent_ep"]:
                counter_task = 0
        reward_pos, counter_task = env.random_stay_task(start_params, counter_task, reward_pos)
        reward_history_tot.append(reward_pos)

        #Reset agent for initial episode
        agent.reset_each_episode(env.agent)

        #Decay of stored_values
        agent.stored_values = agent.stored_values * start_params["stored_values_decay"]

        #Random Guess
        if not any(agent.reward_model == 1) and ep > start_params["latent_ep"]-1:
            possible_food = np.where(env.vector == 2)[0]
            random_guess = np.random.choice(possible_food, 1)
            agent.goal_values[random_guess] += 1

        network_state = "Theta"
        loss_ep = []
        behavior_ep = []
        theta_batch_ep = []
        theta_softmax_ep = []
        SWR_ep = []
        SWR_softmax_ep = []
        stored_values_ep = []
        on_the_fly_values_ep = []
        #----------------------START LOOP STEPS-----------------------------------
        step = 0
        while True: #True
            behavior_ep.append(env.agent)

            #Observations from the world (directly agent position and proximal states)
            observation = env.agent
            agent.proximals = env.agent_available_proximal_states()
            reward = env.check_reward(reward_pos)
            if ep < start_params["latent_ep"]:
                reward = 0


            if agent.reward_model[observation] != reward:
                agent.update_reward_model(observation, reward)
            if reward == 1:
                network_state = "SWR"
                agent.update_reward_location_buffer(observation)
            if env.vector[env.agent] == 2:
                network_state = "SWR"
                if reward == 1:
                    episodio_giusto = 1
                else:
                    episodio_sbagliato = 1


            #Update sensory buffer
            agent.update_sensory_buffer(observation)


            if not (agent_type1 == "No_Theta" and ep > start_params["latent_ep"]-1):
                #Update episodic buffer
                agent.update_episodic_buffer()
            else:
                #Update episodic buffer
                agent.update_episodic_buffer(lesion=True)

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
            
            if step == 16:
                decision_point = 1  #for debugging purposes
            
            if network_state == "Theta" and not (agent_type1 == "No_Theta" and ep > start_params["latent_ep"]-1):
                for pred in range(start_params['theta_compression_factor']):

                    # FIRST THETA HALF - ENCODING
                    ep_sample = agent.sample_from_episodic_buffer(start_params['learning_episodic_batch_size'])
                    [theta_enc[pred], batch_loss, rnn_stateF, rnn_stateB, theta_softmax_enc[pred]] = agent.Encoding(start_params, len_1D_maze,  netF, netB, optimizerF, optimizerB, one_hot_encoder, device, loss_fn, ep=ep)
                    rnn_state = (rnn_stateF[0][:,0,:].unsqueeze(1), rnn_stateF[1][:,0,:].unsqueeze(1))
                    random_state = (rnn_state[0]+torch.randn(rnn_state[0].shape), rnn_state[1]+torch.randn(rnn_state[1].shape))

                    # FIRST THETA PART - Extra - INTEGRATION
                    [theta_int[pred], theta_softmax_int[pred]]  = agent.Integration(theta_enc[pred], theta_softmax_enc[pred], one_hot_encoder)

                    # SECOND THETA PART - FUTURE PREDICTION
                    [theta_pred[pred], theta_softmax_pred[pred]] = agent.Prediction(start_params, netF, netB, device, c_transform, theta_int[pred])

                    # COMPLETE THETA BATCH (Multiple Thetas with Encoding + Prediction parts)
                    theta_batch[pred] = theta_int[pred] + theta_pred[pred]
                    theta_batch_softmax[pred] = theta_softmax_int[pred] + theta_softmax_pred[pred]

                    seq_loss.append(batch_loss)

                theta_batch_ep.append(theta_batch)
                if step == 16:
                  theta_softmax_ep.append(theta_batch_softmax)  #stored only at decision point

                agent.mean_confidence_space(theta_softmax_pred, theta_pred, observation)
                
                # PLANNING
                if any(agent.reward_model == 1):
                    agent.compute_goal_values()
                
                goal_state = np.argmax(agent.goal_values)
                agent.theta_evaluation_goal(theta_pred, theta_softmax_pred, goal_state)


                SWR_ep = []
                SWR_softmax_ep = []

                    
            elif network_state == "SWR" and not (agent_type1 == "No_SWR" and ep > start_params["latent_ep"]-1):

                n_swr = start_params["SWR_compression_factor"]
                if ep < start_params["latent_ep"]:
                    n_swr = 30

                for pred in range(n_swr):

                    # FIRST SWR HALF - ENCODING
                    [SWR_enc, batch_loss_SWR, rnn_stateF, rnn_stateB, SWR_softmax_enc] = agent.Encoding(start_params, len_1D_maze,  netF, netB, optimizerF, optimizerB, one_hot_encoder, device, loss_fn, SWR=True, ep=ep)
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
                
                #print(batch_loss_SWR)
                SWR_ep.append(SWR_batch)
                SWR_softmax_ep.append(SWR_softmax_batch)

                
            #----------------------FINISH LOOP SEQUENCES-----------------------------------

            if step == 16:
                decision_point = 1  #for debugging purposes

            #Behavior
            agent.integrate_state_values(agent_type2)

            if ep > start_params["latent_ep"]-1:
                next_state = agent.behavior_policy_values("state_values")
            else:
                next_state = agent.behavior_pseudo_random_policy()

            #Environment Step
            env.set_agent_position(next_state)

            #End Conditions
            if env.vector[observation] == 2:
                break

            step += 1
            loss_ep.append(np.mean(seq_loss))
            stored_values_ep.append(agent.stored_values)
            on_the_fly_values_ep.append(agent.plan_values)
            #----------------------FINISH LOOP STEPS-----------------------------------

        #SELECT MEASURES TO STORE (for each agent)   
        loss_tot.append(np.mean(loss_ep))
        behavior_tot.append(behavior_ep)
        theta_batch_tot.append(theta_batch_ep)
        theta_softmax_tot.append(theta_softmax_ep)
        SWR_tot.append(SWR_ep)
        #SWR_softmax_tot.append(SWR_softmax_ep)
        stored_values_tot.append(stored_values_ep)
        on_the_fly_values_tot.append(on_the_fly_values_ep)
        

        #print(step)
        print(f"S{ml} - Epoch {ep + 1} loss: {np.mean(loss_ep)}")
        #----------------------FINISH LOOP EPISODES-----------------------------------

    # COLLECT MEASURES (across agents)
    #behavior_loop.append(behavior_tot)
    #reward_history_loop.append(reward_history_tot)
    #loss_loop.append(loss_tot)


#determine environmental condition
if start_params["reward_change_min"] > start_params["n_ep"]:
    env_condition = "_L_"
else:
    env_condition = "_H_"

#SAVE
import os

os.chdir('./C3_produced_data')

id_label = "_" + str(agent_type1) + "_" + str(agent_type2) + str(env_condition) + "SINGLE_AGENT"

with open("loss_tot"+id_label+".txt", "wb") as fp:   #Loss
    pickle.dump(loss_tot, fp)

with open("behavior_tot"+id_label+".txt", "wb") as fp:   #Bahavior
    pickle.dump(behavior_tot, fp)

with open("reward_history_tot"+id_label+".txt", "wb") as fp:   #Reward history
    pickle.dump(reward_history_tot, fp)

with open("theta_batch_tot"+id_label+".txt", "wb") as fp:   #theta_batch_tot
    pickle.dump(theta_batch_tot, fp)

with open("theta_softmax_tot"+id_label+".txt", "wb") as fp:   #theta_softmax_tot
    pickle.dump(theta_softmax_tot, fp)

with open("SWR_tot"+id_label+".txt", "wb") as fp:   #SWR_tot
    pickle.dump(SWR_tot, fp)
  
with open("stored_values_tot"+id_label+".txt", "wb") as fp:   #Stored Values
    pickle.dump(stored_values_tot, fp)
  
with open("on_the_fly_values_tot"+id_label+".txt", "wb") as fp:   #on_the_fly_values_tot
    pickle.dump(on_the_fly_values_tot, fp)
    
os.chdir("..")


#Quick plot at the end
plt.plot(loss_tot)
plt.title('LSTM Loss')
plt.xlabel('Episode')
plt.ylabel('Loss')
plt.show()
import pickle

start_params = {
    #loop   #episodes = latent learning episodes +  task episodes
    "n_ep" : 5,   #set to 350 after check functioning with 5 ep
    "latent_ep" : 5,      #set to 100 after check functioning with 5 ep
    "n_agents" : 1,

    #task
    "reward_change_min" : 50,     #High res: 50      Low res: 1000,
    "reward_change_max" : 80,     #High res: 80       Low res: 1001,

    #agent
    "agent_starting_position" : 192,  #192 is the state at the bottom of the central arm in the 8arm_maze
    "sensory_buffer_size" : 6,
    "episodic_buffer_size" : 6000,      
    "reward_location_buffer_size" : 1,
    "stored_values_decay" : 0.99,

    #model
    "theta_encoding_len" : 5,
    "theta_prediction_len" : 12,             #High res: 12     Low res: 10
    "theta_compression_factor" : 10,          #High res : 10          Low res: 5
    "learning_episodic_batch_size" : 1,
    "SWR_prediction_len" : 10,               #High res: 10    Low res: 8
    "SWR_compression_factor" : 10,            #High res : 10         Low res: 5
    "SWR_reward_episodic_max_batch_size" : 10,
    "SWR_learning_episodic_batch_size" : 16,

    #latent learning
    "theta_prediction_len_latent" : 10,
    "theta_compression_factor_latent" : 5,
    "SWR_prediction_len_latent" : 8,
    "SWR_compression_factor_latent" : 5,

    #LSTM
    "hidden_units" : 64,
    "layers_num" : 1,
    "dropout_prob" : 0.3,
    "learning_rate" : 0.00001,
    "weight_decay" : 0.00001,

}

with open('C3_start_params.pickle', 'wb') as handle:
    pickle.dump(start_params, handle)
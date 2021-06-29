import pickle

start_params = {
    #loop
    "n_ep" : 5, #set to 200 after check functioning with 5 ep

    #task
    "reward_change_min" : 600,
    "reward_change_max" : 610,
    "latent_ep" : 0,

    #agent
    "agent_starting_position" : 0,
    "sensory_buffer_size" : 6,
    "episodic_buffer_size" : 1000,      
    "reward_location_buffer_size" : 1,

    #sequential model
    "theta_encoding_len" : 5,
    "theta_prediction_len" : 12,
    "theta_compression_factor" : 5,
    "learning_episodic_batch_size" : 1,
    "SWR_prediction_len" : 12,
    "SWR_compression_factor" : 5,
    "SWR_reward_episodic_max_batch_size" : 10,
    "SWR_learning_episodic_batch_size" : 120,

    #LSTM
    "hidden_units" : 64,
    "layers_num" : 1,
    "dropout_prob" : 0.3,
    "learning_rate" : 0.005,
    "weight_decay" : 0.00001,

}

with open('C1_start_params.pickle', 'wb') as handle:
    pickle.dump(start_params, handle)
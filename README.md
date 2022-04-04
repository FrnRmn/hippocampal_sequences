--------------------------------------------------
# OVERVIEW

Implementation of a master thesis at University of Padova. Student: Francesco Romandini. Spervisor: Alberto Testolin.

Title: **"A computational investigation on the role of hippocampal sequences in learning spatial representations and informing goal-directed behavior."**

For more info: contact me via email at francescoromandini@gmail.com.

The folder contains the code used to perform simulations in four different contexts.
Some files are shared by all simulations: "agents_dual.py", "environments.py", "lstm_model.py", "utils.py".
Other files are specific for a certain context and are distinguished by a C* mark (e.g., for context 1 -> C1).



--------------------------------------------------
# GETTING STARTED

Before each simulation, "C*_set_parameters.py" needs to be run after the insertion of desired parameters.
It will be created a file called "C*_start_params.pickle" containing parameters data.

Moreover, depending on the context it is possible to specify some extra parameters directly inside the simulation code: "C*_simulation.py".
More specifically: factors needs to be specificed before class objects initialization, measures to be saved needs to be specified before the finish of episodes loop.



--------------------------------------------------
# PERFORM SIMULATIONS

The simulation can be performed running the "C*_simulation.py" file.
It will be printed a log with the average loss measure for each episode.
At the end, files will be directly saved to a folder named "C*_produced_data".


--------------------------------------------------
# SOME ANTICIPATIONS

### 1) A neural network LSTM model generates simulated hippocampal sequences which can inform behavior in a RL framework
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/93c6eefd8a0224f09caedbb14645c145cff7a0af/images/loop_RL.gif)

### 2) Three different environments
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/93c6eefd8a0224f09caedbb14645c145cff7a0af/images/environments_smaller.jpg)

### 3) Backward skewing of place field
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/5ee61941df4e59138f36a347918cf26c31ebd2b3/images/FigRes1.png)

### 4) Place fields and Grid fields simulation
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/5ee61941df4e59138f36a347918cf26c31ebd2b3/images/FigRes2.png)

### 5) The agent learn to perform a spatial decision task between 8 choice arms
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/5ee61941df4e59138f36a347918cf26c31ebd2b3/images/behavior.gif)

### 6) Theta sequences at decision point represent alternative future paths
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/5ee61941df4e59138f36a347918cf26c31ebd2b3/images/theta_part.gif)

### 7) SWR sequences at reward location represent the path that led the agent to reward in backward order
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/5ee61941df4e59138f36a347918cf26c31ebd2b3/images/swr.gif)

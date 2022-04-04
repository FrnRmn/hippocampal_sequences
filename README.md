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
*Schematic representation of the classical reinforcement learning loop. The environment pass 
state and reward information to the agent at time t. At the same time the agent performs an action 
that produce effects on the environment. As a response, the environment outputs another pair of state 
and reward measures at the next state, and the loop can be repeated.  The main functions of theta sequences are highlighted in blue (T1 
is the encoding function, T2 is the future path evaluation function). Main functions of SWR sequences 
are highlighted in green (S1 is the training function of the Bi-LSTM network, S2 is the past rewarded 
path valuation function).*
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/93c6eefd8a0224f09caedbb14645c145cff7a0af/images/loop_RL.gif)
<br>

### 2) Three different simulation environments
*(A) Corridor environment. 30 squared discrete 
states are positioned in a straight line from left to right. In this example the agent starts from the left 
extremity, moves right with a directional policy and aim to reach the right extremity that concludes 
the episode. (B) Open arena environment. 100 squared discrete states are positioned in a 10x10 matrix 
structure. In this example the agent moves around the map with a random walk policy. (C) 8-arm 
maze environment. At the bottom it is represented the original environmental shape. At the top it is 
represented the environmental shape adapted for the implementation. A starting arm of 17 squared 
discrete states is connected with other eight choice arms of 11 squared discrete states. Each choice 
arm holds a feeder square. One arm for each episode holds a reward in its feeder square*
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/be08525100d42461006141ef5b34f7b6bfeb484e/images/environments_smaller.jpg)
<br>

### 3) Backward skewing of place field in environment A
*(A) Bi-LSTM cross-entropy loss function plotted in respect of training 
episodes. In red is shown the random walk policy agent. In Blue is shown the directional policy agent. 
(B) Simulated successor matrix belonging to the directional agent. The size is 30x30, since the 
environment has 30 states. (C) Normalized activity of the simulated place cell tuned to location 19 in 
function of the 30 environmental locations. In red is shown the random walk policy agent. In Blue is 
shown the directional policy agent. (D) Real hippocampal neuron activity recorded during the first lap 
(red) and the last lap (blue) of a directional running task. Image taken from (Mehta et al., 2000).*
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/5ee61941df4e59138f36a347918cf26c31ebd2b3/images/FigRes1.png)
<br>

### 4) Place fields and Grid fields simulation in environment B
*(A) Example of place cells recorded from the rat hippocampus CA3 
area. On the left: trajectories and neuron spikes in red. On the right: rate maps. Image taken from 
(Fyhn et al., 2007). (B) Simulated place fields representing normalized activity of simulate place cells in 
the open arena environment. (C) Example of grid cells recorded from rat medial entorhinal cortex
(MEC). On the left: trajectories and neuron spikes in red. On the right: rate maps. Image taken from 
(Giocomo, Moser, & Moser, 2011). (D) Simulated grid fields representing the values of eigenvectors 
components computed from the eigen-decomposition of the simulated successor matrix, in an open 
arena environment.*
![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/5ee61941df4e59138f36a347918cf26c31ebd2b3/images/FigRes2.png)
<br>

### 5) The agent learn to perform a spatial decision task between 8 choice arms in environment C
*The position of the agent is represented by the orange dot. The decision point is highlighted in violet. The eight 
possible reward points are highlighted in yellow.*<br>


![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/5ee61941df4e59138f36a347918cf26c31ebd2b3/images/behavior.gif)
<br>

### 6) Theta sequences at decision point represent alternative future paths in environment C
*Examples of theta sequences representations at decision point. The darkest blue color indicates the beginning of the sequence while the lightest blue color indicates the end of the sequence.*<br>


![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/5ee61941df4e59138f36a347918cf26c31ebd2b3/images/theta_part.gif)
<br>

### 7) SWR sequences at reward location represent the path that led the agent to reward in backward order in environment C
*Examples of SWR sequences representations at reward location. The darkest green color indicates the beginning of the sequence while the lightest green color indicates the end of the sequence.*<br>


![alt-text](https://github.com/FrnRmn/hippocampal_sequences/blob/5ee61941df4e59138f36a347918cf26c31ebd2b3/images/swr.gif)
<br>

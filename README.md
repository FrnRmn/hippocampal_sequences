# OVERVIEW
--------------------------------------------------
Implementation of a master thesis at University of Padova. Student: Francesco Romandini. Spervisor: Alberto Testolin.

Title: **"A computational investigation on the role of hippocampal sequences in learning spatial representations and informing goal-directed behavior."**

For more info: __link available soon__ or contact me via email at francescoromandini@gmail.com.

The folder contains the code used to perform simulations in four different contexts.
Some files are shared by all simulations: "agents_dual.py", "environments.py", "lstm_model.py", "utils.py".
Other files are specific for a certain context and are distinguished by a C* mark (e.g., for context 1 -> C1).




# GETTING STARTED
--------------------------------------------------
Before each simulation, "C*_set_parameters.py" needs to be run after the insertion of desired parameters.
It will be created a file called "C*_start_params.pickle" containing parameters data.

Moreover, depending on the context it is possible to specify some extra parameters directly inside the simulation code: "C*_simulation.py".
More specifically: factors needs to be specificed before class objects initialization, measures to be saved needs to be specified before the finish of episodes loop.




# PERFORM SIMULATIONS
--------------------------------------------------
The simulation can be performed running the "C*_simulation.py" file.
It will be printed a log with the average loss measure for each episode.
At the end, files will be directly saved to a folder named "C*_produced_data".

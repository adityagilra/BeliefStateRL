# BeliefStateRL

These simulations were run on linux within a conda environment whose exported list of dependencies is in environment.yml (also a pip export in requirements.txt).
However these (environment.yml and requirements.txt) have a lot of packages which are not needed for this repo.   
The packages below (and their dependencies) should be sufficient (version numbers are those that I used, but nearby ones should work as well):  
python=3.6.10  
pip:  
 gym==0.15.7  
 numpy==1.19.0  
 matplotlib==3.1.1  
 scipy==1.5.1  
  
## To run simulations of the Simple and BeliefState RL agents with pre-fitted parameters to the experimental data:  
First, `cd` to the BeliefStateRL directory.  
Create a directory to store simulation data (only once)  
`mkdir simulation_data`  
`python BeliefHistoryTabularRLSimulate.py`  
This will plot results as well.  

You can select the type of agent (basic vs belief) and number of parameters (2 and 4 resp. were used in the paper, though fits were obtained for more) by modifying the script:  
```python
    ############## choose / uncomment one of the agents below! #################
    agent_type='belief'
    #agent_type='basic'

    if agent_type == 'basic':
        # choose one of the below
        num_params_to_fit = 2 # for both basic and belief RL
        #num_params_to_fit = 3 # for both basic and belief RL
    else:
        # choose one of the below
        #num_params_to_fit = 2 # for both basic and belief RL
        #num_params_to_fit = 3 # for both basic and belief RL
        num_params_to_fit = 4 # only for belief RL
        
    # choose whether ACC is inhibited or not
    #ACC_off = True
    ACC_off = False
```

You can also choose whether to simulate with the parameters that fit either the primary dataset used for Fig. 1 in the paper with ACC not silenced by setting `new_data=True`,
 or the dataset with ACC silenced vs not silenced by setting `new_data=False` and ACC_off=True or False.  
 See Table 1 in the Methods section of the paper for these parameters.  
```python
    # whether to use parameters obtained by fitting to:
    #  old (has ACC-on/control and ACC-off/exp) data,
    #  or new (behaviour+neural w/ only ACC-on) data.
    new_data = True
    #new_data = False
```

## The experimental data is in the folder `experiment_data`.  
Some simple plots of the data can be made using:  
`python exp_data_analysis.py`  

To plot simulation results later, use
`plot_simulation_data.py`
and
`plot_exp_sim_data.py`
after editing the filename of the simulation data in this script.

## The hierarchical task used in the the experiments to train the mice is transcribed as an OpenAI Gym environment in the directory `gym_tasks/envs/`.
 See `BeliefStateRLSimulate.py` on how to import and `BeliefStateRL.py` on how to use the environment.  

## To fit the experimental data:  
`python fit.py`  

Again set the type of agent, number of parameters to fit, etc. in `fit.py`.  
What these parameters do can be seen as comments in BeliefStateRLSimulate.py, more formally in Methods sections of the paper -- the definitive version is as coded into the agent in BeliefStateRL.py.  

For model selection i.e. comparing basic vs belief models: Set `k_validation=5` to do 5-fold cross validation -- fitting parameters on 4/5th of the data and testing on 1/5th of the data
 using the specified model/agent (using only 1 seed i.e. not averaging over multiple seeds).  
For obtaining fitted parameters for the best fit: Set `k_validation=1` to fit on the full data averaging RMSE of the agent across 5 seeds.  

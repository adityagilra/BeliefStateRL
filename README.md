# BeliefStateRL
  
This is the code and experimental data to simulate and fit the Basic RL and Belief State RL models from version 2 of the [bioRxiv preprint Cole, et al., Prediction-error signals in anterior cingulate cortex drive task-switching, 2022](https://www.biorxiv.org/content/10.1101/2022.11.27.518096v2).  
  
## Prerequisites
These simulations were run on linux within a conda environment whose exported list of dependencies is in environment.yml (also a pip export in requirements.txt).
However these (environment.yml and requirements.txt) have a lot of packages which are not needed for this repo.   
The packages below (and their dependencies) should be sufficient (version numbers are those that I used, but nearby ones should work as well):  
python=3.6.10  
pip install <package_name>:  
 gym==0.15.7  
 numpy==1.19.0  
 matplotlib==3.1.1  
 scipy==1.5.1  
  
## To run simulations of the Simple and BeliefState RL agents with pre-fitted parameters to the experimental data:  
First, `cd` to the BeliefStateRL directory.  
Create a directory to store simulation data (only once)  
`mkdir simulation_data`  

### Run simulation of RL agent with pre-fitted parameters:  
`python BeliefHistoryTabularRLSimulate.py <belief|basic> <True|False> <0|1|2>`  
belief|basic are to use BeliefStateRL or BasicRL agent.  
True|False simulates with ACC being off i.e. inhibited (True) versus on (False).  
0|1|2 refers to using different datasets (use 0 or 2; 1 is obsolete since it is incorporated into 2):  
 0 is a dataset on older task with ACC on and off conditions,  
           used for ACC on vs off switching times comparisons  
 1 is a dataset on older task with only ACC on having 13 sessions, this is the dataset with neural recordings as well  
          used for mismatch of context error signals between fast vs slow switches, on old task only  
          This dataset 1 is obsolete for simulation purposes, as it is incorporated into dataset 2 as below.  
 2 is a dataset, with only ACC on condition, of 4 sessions on newer task (having 1st cue after V2O as unrewarded V2),  
            prepended to the above 13 sessions of dataset 1 (on older task having 1st cue after V2O as unrewarded V1)  
          used for comparing Belief State RL vs Basic RL, on new and old tasks in 4:13 ratio.  
  
This script does not fit params,  
 rather it simulates the agent setting params which were fitted already using fit.py for each of the specifications above.  
These params are already hardcoded into BeliefHistoryTabularRLSimulate.py for various specifications,  
    so no need to run fit.py unless you want to re-fit.  
 See Table 1 in the Methods section of the paper for the model parameters used for these simulations, obtained after fitting (see further below for how to fit).  
_____
### Reproducing specific figures:  
Figure numbers are for [v2 of the pre-print](https://www.biorxiv.org/content/10.1101/2022.11.27.518096v2).  
  
For Fig 1H and Suppl. Fig 3C, run (ACC_off=False and new_data=2):  
`python BeliefHistoryTabularRLSimulate.py belief False 2`  
for BeliefStateRL,  
or  
`python BeliefHistoryTabularRLSimulate.py basic False 2`  
for BasicRL.  
_____
All runs of BeliefHistoryTabularRLSimulate.py will show plots in 2 batches (2 plt.show()s) at the end of the simulations,  
 but you can re-plot the saved files later using `python plot_simulation_data.py` (only simulated data)  
  or `python plot_exp_sim_data.py` (sim data overlaid on exp data).  
Comment/uncomment the relevant lines in __main__ there, e.g.:  
```python  
load_plot_simdata('simulation_data/simdata_belief_numparams4_nolearnwithexplore_ACCcontrol_newdata2',seeds)  
```  
_____
For Fig 6C, ensure that you have earlier run:  
`python BeliefHistoryTabularRLSimulate.py belief False 2`  
then uncomment in plot_simulation_data.py:  
```python  
load_plot_simdata('simulation_data/simdata_belief_numparams4_nolearnwithexplore_ACCcontrol_newdata2',seeds=[2])  
```  
comment other plotting lines, and run.  
_____
For Suppl Fig 2D and Suppl Fig 3F (uses older dataset, hence 0, that has data for ACC off and on), run  
`python BeliefHistoryTabularRLSimulate.py belief True 0`  
and  
`python BeliefHistoryTabularRLSimulate.py belief False 0`  
then comment/uncomment the plotting lines for Suppl Fig 2D in plot_simulation_data.py and run.  
  
### Changing the number of (pre-fitted) parameters used in simulating the model  
You can modify, in BeliefHistoryTabularRLSimulate.py, the number of pre-fitted parameters (default 2 for basic and 4 for belief were used in the paper and set here) by commenting/uncommenting lines 416 to 424 of __main__:  
```python
    ############## choose / uncomment one of the agents below! #################
    if agent_type == 'basic':
        # choose one of the below
        num_params_to_fit = 2 # for both basic and belief RL
        #num_params_to_fit = 3 # for both basic and belief RL
    elif agent_type == 'belief':
        # choose one of the below
        #num_params_to_fit = 2 # for both basic and belief RL
        #num_params_to_fit = 3 # for both basic and belief RL
        num_params_to_fit = 4 # only for belief RL
```
We obtained fits for all these num_params_to_fit - see below for how to fit parameters. These fitted values of parameters for these num_params_to_fit, are hard-coded into BeliefHistoryTabularRLSimulate.py

## The experimental data is in the folder `experiment_data`  
Some simple plots of the data can be made using:  
`python exp_data_analysis.py`  

To plot simulation results later, use
`plot_simulation_data.py`
and
`plot_exp_sim_data.py`
after editing the filename of the simulation data in __main__ in this script.

## Experimental task as OpenAI Gym environment:  
The hierarchical task used in the the experiments to train the mice is transcribed as an OpenAI Gym environment in the directory `gym_tasks/envs/`.  
 See `BeliefStateRLSimulate.py` on how to import and `BeliefStateRL.py` on how to use the environment.  

## To fit the experimental data:  
`python fit.py`  
  
Set the type of agent, number of parameters to fit, etc. in `fit.py`.  
What these parameters do can be seen as comments in BeliefHistoryTabularRLSimulate.py, more formally in Methods sections of the paper -- the definitive version is as coded into the agent in BeliefHistoryTabularRL.py.  
Here too, you need to select `new_data=0`/`1`/`2` and also whether `ACC_off=True`/`False` as above.  
  
For model selection i.e. comparing basic vs belief models: Set `k_validation=5` to do 5-fold cross validation -- fitting parameters on 4/5th of the data and testing on 1/5th of the data
 using the specified model/agent (uses 5 seeds and if `new_data==2` then first 2 seeds are run on new task and rest 3 on older task).  
For obtaining fitted parameters for the best fit on the full dataset: Set `k_validation=1` to fit on the full data averaging RMSE of the agent across 5 seeds.  

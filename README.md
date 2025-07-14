# BatteryRLThesis

This repository contains the code, data structure, and experiments for my econometrics master's thesis on optimizing behind-the-meter battery storage for Dutch SMEs using forecast-based reinforcement learning.

## Folder structure

The 'data' folder contains all necessary data.  

The 'model' folder contains the stored, trained models that can be loaded into "ThesisNotebook" for easy evaluation of trained models.  

The 'notebooks' folder contains jupiter notebooks which are working files used throughout the thesis process. 'ThesisDataCollection', 'ThesisModelsClean' and 'ThesisBattery' are old notebooks, and most code that is used is integrated in 'ThesisNotebook'. 

The 'scripts' folder contains different python files with functions that are used by the notebooks. In this folder, a python file containing the BatteryEnvironment (BatteryEnv.py) can be found, as well as helpers.py containing all helper functions, including hyperparameter tuning function. DQN.py, PPOc.py, PPOd.py and SAC.py contain their respective algorighms and are also used in 'ThesisNotebook'. The code for setting up the energy price forecasting models can be found in EPFmodels.py. 
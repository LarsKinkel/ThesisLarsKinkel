# BatteryRLThesis

This repository contains the code, data pipeline, and experiments for my econometrics master’s thesis:  
**“Optimizing behind-the-meter battery storage for Dutch SMEs participating in the day-ahead electricity market”**  
This repo is made public to enhance transparency and reproducibility of results. 

---

## Repository Structure

```
BatteryRLThesis/
├── data/            # Raw and processed datasets (price, PV, demand, weather)
├── models/          # Trained RL models under different specifications (DQN, PPO, SAC)
├── notebooks/       # Jupyter notebooks for data processing, training, evaluation
├── scripts/         # Python modules for environment, forecasting, agents, utilities
├── .gitignore       # Specifies untracked files to ignore in Git
└── README.md        # This file in md
```
---

## Folder Descriptions

- **`data/`**  
  Contains the input data, including electricity price, market variables, and weather time series used in training and evaluation.

- **`models/`**  
  Trained reinforcement learning models (DQN, PPO, SAC), which can be loaded for evaluation.

- **`notebooks/`**  
  Contains the main and auxiliary Jupyter notebooks:
  - `ThesisNotebook.ipynb`: Final evaluation notebook combining forecasting and control.
  Run this notebook to see how the results of the thesis are obtained. The notebook as it is in this repo will use only stored models and data, and will not train models or perform hyperparameter tuning, so running all cells will only take a few minutes. Note that the output of the cells is also visible already in `ThesisNotebook.ipynb`

- **`scripts/`**  
  Modular Python scripts with key functionality:
  - `BatteryEnv.py`: Custom OpenAI Gym-compatible battery environment used for training agents.
  - `helpers.py`: Utility functions, data processing, and hyperparameter tuning support.
  - `EPFmodels.py`: Implements energy price forecasting models (e.g., AR, LEAR).
  - `DQN.py`, `PPOc.py`, `PPOd.py`, `SAC.py`: Contain the respective reinforcement learning agents.  
  See through these files to get insight on how methods are implemented 

---

## Getting Started

1. Clone the repository in a terminal:  
   git clone https://github.com/LarsKinkel/ThesisLarsKinkel.git  
   cd ThesisLarsKinkel  

2. Set up a python environment (recommended Python 3.9+):  
Using conda:  
    conda create -n battery-thesis python=3.10  
    conda activate battery-thesis  

3. Install the requirements from the requirements.txt file  
    pip install -r requirements.txt  

4. Open Jupyter or VSCode and run the notebook:  
    notebooks/ThesisNotebook.ipynb  



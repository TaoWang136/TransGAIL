# TransGAIL
## Environment
follow_env_v1.py is my custom environment. You need to create your own environment based on GYM.
## Data
There are four locations, each of which filters out vehicles that take evasive action. Each file contains data on the interaction between the ego vehicle and the surrounding vehicles.
## Method
Run train_yield.py to extract the expert's evasion policy.<br>
Run test_yield.py to test the expert's evasion policy.<br>
The settings of surrounding vehicles follow the original dataset.

#!/bin/bash

# export PYTHONPATH=$PYTHONPATH:../evaluate/
# export PYTHONPATH=$PYTHONPATH:../preprocess/
# 
# while true; do
#     current_epoch=$(date +%Y_%m_%d_%H_%M)
#     
#     printf "$current_epoch\n"
# 
#     #/usr/local/bin/python3 /Users/simon/Documents/code/nodejs/master-project/mp_data_analysis/analysis/learn.py
#     #> ../outputs/learn_$current_epoch.log 2>&1 3>&1 4>&1 5>&1 6>&1 7>&1 8>&1 9>&1
#     
#     # https://unix.stackexchange.com/questions/190773/redirect-script-output-to-dev-tty1-and-also-capture-output-to-file
#     /usr/local/bin/python3 -u /Users/simon/Documents/code/nodejs/master-project/mp_data_analysis/analysis/learn.py | tee /dev/tty1 ../outputs/learn_$current_epoch.log
#     
#     sleep 1
#     
# done

source activate mp

current_epoch=$(date +%Y_%m_%d_%H_%M)

# Use PYTHONUNBUFFERED: Force stdin, stdout and stderr to be totally unbuffered.
nohup python -u ./learn.py &> ../outputs/bg_learn_$current_epoch.log &

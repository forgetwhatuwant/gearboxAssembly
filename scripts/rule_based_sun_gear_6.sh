#!/bin/bash
# filepath: gearboxAssembly/scripts/rule_based_sun_gear_6.sh

# 1. Activate Conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_isaaclab

# Define the command to run
CMD="python scripts/rule_based_agent.py --task=Template-Galaxea-Sun-Gear-6-v0 --enable_cameras"

# Define runtime for each iteration (seconds)
# Longer runtime for Stage 2 (more complex assembly)
RUNTIME=180  

# Loop to run the command repeatedly
while true; do
  echo "Starting command: $CMD"
  
  # Run the command with a timeout
  timeout $RUNTIME $CMD
  
  # Check exit status
  EXIT_STATUS=$?
  
  if [ $EXIT_STATUS -eq 124 ]; then
    echo "Command timed out after $RUNTIME seconds. Restarting..."
  else
    echo "Command exited with status $EXIT_STATUS. Restarting..."
  fi
  
  # Optional: brief pause before restart to avoid rapid loops if crashing immediately
  sleep 1
done

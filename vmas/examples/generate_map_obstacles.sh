#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting map generation..."

# Example Command 1: Generate 20 training maps
echo "Generating training maps (20)..."
python3  map_generator.py --num 200 --output ../train_maps_1_obstacle --level 1 --type obstacle --path 0


# Example Command 2: Generate 5 validation maps
echo "Generating validation maps (5)..."
python3  map_generator.py --num 200 --output ../train_maps_2_obstacle --level 2 --type obstacle --path 0


# Example Command 3: Generate 10 test maps with different settings (example)
echo "Generating test maps (10) with specific seed..."
python3  map_generator.py --num 200 --output ../train_maps_3_obstacle --level 3 --type obstacle  --path 0

python3  map_generator.py --num 200 --output ../train_maps_4_obstacle --level 4 --type obstacle --path 0


python3  map_generator.py --num 200 --output ../train_maps_5_obstacle --level 5 --type obstacle --path 0



# Add more commands as needed below
# echo "Generating another set..."
# python3 map_generator.py --num 50 --output ../large_maps --complexity high

echo "Map generation finished successfully."
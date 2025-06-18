#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting map generation..."


python3  map_generator.py --num 200 --output ../train_maps_0_clutter --level 0 --type clutter 

# Example Command 1: Generate 20 training maps
echo "Generating training maps (20)..."
python3  map_generator.py --num 200 --output ../train_maps_1_clutter --level 1 --type clutter 


# Example Command 2: Generate 5 validation maps
echo "Generating validation maps (5)..."
python3  map_generator.py --num 200 --output ../train_maps_2_clutter --level 2 --type clutter 


# Example Command 3: Generate 10 test maps with different settings (example)
echo "Generating test maps (10) with specific seed..."
python3  map_generator.py --num 200 --output ../train_maps_3_clutter --level 3 --type clutter

python3  map_generator.py --num 200 --output ../train_maps_4_clutter --level 4 --type clutter 


python3  map_generator.py --num 200 --output ../train_maps_5_clutter --level 5 --type clutter 



# Add more commands as needed below
# echo "Generating another set..."
# python3 map_generator.py --num 50 --output ../large_maps --complexity high

echo "Map generation finished successfully."
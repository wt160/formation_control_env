#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting map generation..."


python3  map_generator.py --num 500 --size 128 --output ../train_maps_0_test --level 0 --type test 

# Example Command 1: Generate 20 training maps
echo "Generating training maps (20)..."
python3  map_generator.py --num 500 --size 128 --output ../train_maps_1_test --level 1 --type test 


# Example Command 2: Generate 5 validation maps
echo "Generating validation maps (5)..."
python3  map_generator.py --num 500 --size 128 --output ../train_maps_2_test --level 2 --type test 


# Example Command 3: Generate 10 test maps with different settings (example)
echo "Generating test maps (10) with specific seed..."
python3  map_generator.py --num 500 --size 128 --output ../train_maps_3_test --level 3 --type test

python3  map_generator.py --num 500 --size 128 --output ../train_maps_4_test --level 4 --type test 


python3  map_generator.py --num 500 --size 128 --output ../train_maps_5_test --level 5 --type test



# Add more commands as needed below
# echo "Generating another set..."
# python3 map_generator.py --num 50 --output ../large_maps --complexity high

echo "Map generation finished successfully."
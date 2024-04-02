# This script generates images with repetiive objects using Blender.
cd image_generation/
blender --background --python render_images.py -- --num_images 1 --shape_color_combos_json data/CoGenT_A.json --min_dist 0.3
cd ..
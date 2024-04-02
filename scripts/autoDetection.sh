# Script to run the autoDetection module
cd autoDetection/
python autosim.py -q ../'experimental results'/'repetitive objetcs'/output/images/CLEVR_new_000000.png -a im3_sub.png -l "build/libautosim.so" -m 0.8 -n 1000 -i 0 -r 4 -w 800
cd ..
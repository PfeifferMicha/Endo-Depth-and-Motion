
# EXAMPLE:
python3 apps/tracking_ours/__main__.py -d cuda:0 --input_scene_directory data/daVinci/2019_10_10/ -o data/daVinci/2019_10_10/tracking --no_vis_3d
# Replace .pkl file with output from line above!
python apps/volumetric_fusion/__main__.py -i data/daVinci/2019_10_10/tracking/20240916-144213.pkl -o data/daVinci/2019_10_10/

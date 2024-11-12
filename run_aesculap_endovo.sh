#!/bin/bash

#set -x

source ~/.bashrc

#for patient in 2021_04_19 2021_04_26 2021_05_27 2021_06_07 2021_06_15 2021_06_23 2021_06_30 2021_07_14 2021_08_31
#for patient in 2021_04_26 2021_05_27 2021_06_07 2021_06_15 2021_06_23 2021_06_30 2021_07_14 2021_08_31
#for patient in 2021_06_07 2021_08_31
#for patient in 2021_04_14 2021_04_19 2021_04_26 2021_05_27 2021_06_07 2021_06_15 2021_06_23 2021_06_30 2021_07_14 2021_08_31

for patient in 2021_04_19
#for patient in 2021_04_14 2021_04_19 2021_04_26 2021_05_27 2021_06_07 2021_06_15 2021_06_23 2021_06_30 2021_07_14 2021_08_31
do
    patient_orig_path=/mnt/ceph/tco/TCO-All/Projects/Mediassist3/recordings/Human/AesculapARAILIS/$patient/Rectified_shortened_extreme
    patient_output_path=~/Projects/IntraoperativeReconstruction/Endo-Depth-and-Motion/data/Aesculap/$patient/

    if [ -d "$patient_orig_path" ]; then
        echo "==================================================================="
        echo "Parsing data in: $patient_orig_path"
        echo "Storing in: $patient_output_path"
        
        mkdir $patient_output_path -p

        cd $patient_output_path
        ln -s -n $patient_orig_path color

        echo "================================"
        echo "Depth reconstruction:"
        echo "================================"
        # Depth estimation
        #cd ~/Projects/IntraoperativeReconstruction/PCWNet
        #conda init
        #conda activate pcw_net
        #python3 apply.py --dataset Aesculap --model gwcnet-gc --loadckpt ceph_models/PCWNet_sceneflow_pretrain.ckpt --datapath $patient_output_path/color --outpath $patient_output_path/depth

        #cd ~/Projects/IntraoperativeReconstruction/Depth-Anything-V2
        #conda init
        #conda activate depth_anything
        #python3 run.py --encoder vitb --img-path $patient_output_path/color --outdir $patient_output_path/depth_anything_v2/ --pred-only --save-raw --rescale $patient_output_path/depth

        echo "================================"
        echo "Tracking:"
        echo "================================"
        #cd ~/Projects/IntraoperativeReconstruction/Endo-Depth-and-Motion
        #conda init
        #conda activate endo_depth_and_motion
        #
        #python3 apps/tracking_ours/__main__.py -d cuda:0 --input_scene_directory $patient_output_path -o $patient_output_path/tracking --no_vis_3d --no_vis_2d

        #tracking_result=$(ls -t $patient_output_path/tracking/20*.pkl | head -1)
        tracking_result=$(ls -t $patient_output_path/endovo/infer_trajectory/*.txt | head -1)
        echo Tracking result: $tracking_result

        echo "================================"
        echo "Volume fusion:"
        echo "================================"
        cd ~/Projects/IntraoperativeReconstruction/Endo-Depth-and-Motion
        conda init
        conda activate endo_depth_and_motion
        python3 apps/volumetric_fusion/__main__.py -i $tracking_result -o data/Aesculap/$patient/ --use_poses_from_endovo
    fi

done

#bin/bash/!
python /home/ns38942/RepresentationLearning/code/ML/top_test.py \
--gpu_id $1 \
--chop_size 4096 \
--slice_size 2048 \
--pkl_dataset_path /home/ns38942/RepresentationLearning/pkl_files/dataset.pkl \
--weight_path /home/ns38942/AiR/results/IQ/weights-OOD_CW.pt \

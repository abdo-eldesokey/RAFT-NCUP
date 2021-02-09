#!/bin/bash
mkdir -p checkpoints
if [ "$HOSTNAME" = "n2018" ] || [ "$HOSTNAME" = "n2017" ]; then
  rsync -avP /proj/gpu-test/users/x_abdel/datasets/flyingthings.tar /scratch/local
  rsync -avP /proj/gpu-test/users/x_abdel/datasets/mpi_sintel.tar /scratch/local
  cd /scratch/local
  mkdir FlyingThings3D
  tar -xvf flyingthings.tar -C FlyingThings3D/
  tar -xvf mpi_sintel.tar
  cd /proj/gpu-test/users/x_abdel/code/RAFT
  ln -s /scratch/local/FlyingThings3D/ datasets/FlyingThings3D
  ln -s /scratch/local/mpi_sintel datasets/Sintel
fi

EXP=raft_nc533_things_ft

python -u train.py \
--name $EXP \
--model raft_nc_dbl \
--load_pretrained models/raft-things.pth \
--stage things \
--validation sintel \
--compressed_ft \
--gpus 0 1 \
--num_steps 100000 \
--batch_size 6 \
--lr 0.000125 \
--image_size 400 720 \
--optimizer adamW \
--scheduler cyclic \
--final_upsampling=NConvUpsampler \
--final_upsampling_scale=4 \
--final_upsampling_use_data_for_guidance=True \
--final_upsampling_channels_to_batch=True \
--final_upsampling_use_residuals=False \
--final_upsampling_est_on_high_res=False \
--interp_net=NConvUNet \
--interp_net_channels_multiplier=2 \
--interp_net_num_downsampling=1 \
--interp_net_data_pooling="conf_based" \
--interp_net_encoder_filter_sz=5 \
--interp_net_decoder_filter_sz=3 \
--interp_net_out_filter_sz=1 \
--interp_net_shared_encoder=True \
--interp_net_use_double_conv=False \
--interp_net_use_bias=False \
--weights_est_net=Simple \
--weights_est_net_num_ch="[64, 32]" \
--weights_est_net_filter_sz="[3, 3, 1]" \
--weights_est_net_dilation="[1, 1, 1]" \

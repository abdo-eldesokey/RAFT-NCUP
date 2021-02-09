#!/bin/bash
mkdir -p checkpoints

EXP=raft_nc_sintel_ft

python -u train.py \
--name $EXP \
--model raft_nc_dbl \
--load_pretrained models/raft-sintel.pth \
--stage sintel \
--validation sintel \
--gpus 0 1 \
--num_steps 50000 \
--batch_size 6 \
--lr 0.000125 \
--image_size 368 768 \
--optimizer adamW \
--scheduler cyclic \
--gamma=0.85 \
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

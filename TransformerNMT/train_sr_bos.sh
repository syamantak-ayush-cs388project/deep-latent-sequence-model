MONO_DATASET='0:./data/sr_bos/train_0.spm32000.txt.pth,,;1:./data/sr_bos/train_1.spm32000.txt.pth,,'
PARA_DATASET='0-1:,./data/sr_bos/dev_X.spm32000.txt.pth,./data/sr_bos/test_X.spm32000.txt.pth'
PRETRAINED='./data/sr_bos/all.256.vec'

python main.py \
--exp_name sr_bos \
--transformer True \
--n_enc_layers 3 \
--n_dec_layers 3 \
--share_enc 2 \
--share_dec 2 \
--share_lang_emb True \
--share_output_emb True \
--emb_dim 256 \
--langs '0,1' \
--n_mono -1 \
--n_para 0 \
--mono_dataset $MONO_DATASET \
--para_dataset $PARA_DATASET \
--mono_directions '0,1' \
--word_shuffle 3 \
--word_dropout 0.1 \
--word_blank 0.2 \
--pretrained_emb $PRETRAINED \
--pretrained_out True \
--lambda_xe_mono 1 \
--otf_num_processes 25 \
--otf_sync_params_every 1000 \
--enc_optimizer adam,lr=0.0001 \
--group_by_size True \
--batch_size 16 \
--epoch_size 10000 \
--stopping_criterion bleu_0_1_valid,20 \
--save_periodic True

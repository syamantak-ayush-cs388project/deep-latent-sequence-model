MONO_DATASET='0:./data/shakespeare/train_0.txt.pth,,;1:./data/shakespeare/train_1.txt.pth,,'
PARA_DATASET='0-1:,./data/shakespeare/dev_X.txt.pth,./data/shakespeare/test_X.txt.pth'
PRETRAINED='./data/shakespeare/all.256.vec'

python main.py \
--exp_name author_imitation \
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
--otf_num_processes 1 \
--otf_sync_params_every 1000 \
--enc_optimizer adam,lr=0.0001 \
--group_by_size True \
--batch_size 16 \
--epoch_size 10000 \
--stopping_criterion bleu_0_1_valid,20 \
--save_periodic True
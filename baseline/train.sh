rm -r results/nmmo
python monobeast.py \
    --total_steps 1000000000 \
    --learning_rate 0.0003 \
    --entropy_coef 0.001 \
    --clip_ratio 0.2 \
    --num_selfplay_team 8 \
    --data_reuse 8 \
    --num_actors 12 \
    --batch_size 512 \
    --unroll_length 32 \
    --savedir ./results \
    --checkpoint_interval 3600 \
    --xpid nmmo

rm -r results/nmmo
python monobeast.py \
    --total_steps 1000000000 \
    --learning_rate 0.0001 \
    --entropy_coef 0.005 \
    --upgo_coef 0.2 \
    --clip_ratio 0.2 \
    --num_selfplay_team 8 \
    --data_reuse 8 \
    --num_actors 4 \
    --batch_size 11 \
    --unroll_length 32 \
    --savedir ./results \
    --checkpoint_interval 600 \
    --restart_actor_interval 18000 \
    --reward_setting phase2 \
    --xpid nmmo \
    $@ \
    #--checkpoint_path ./checkpoints/model_2757376.pt
    
    

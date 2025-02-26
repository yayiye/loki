gamma=0.95
lmbda=0.95
actor_lr=1e-4
critic_lr=1e-3
eps=0.2
n_hiddens=128
python gcc_trainer.py \
    --gamma ${gamma} \
    --lmbda ${lmbda} \
    --actor_lr ${actor_lr} \
    --critic_lr ${critic_lr} \
    --eps ${eps} \
    --n_hiddens ${n_hiddens}

# for((ITER=1; ITER<=3; ITER++))
# do
#     python gcc_trainer.py \
#         --gamma ${gamma} \
#         --lmbda ${lmbda} \
#         --actor_lr ${actor_lr} \
#         --critic_lr ${critic_lr} \
#         --eps ${eps} \
#         --n_hiddens ${n_hiddens}
    
#     actor_lr=$(awk "BEGIN {print${actor_lr} + 1e-4}")
#     critic_lr=$(awk "BEGIN {print${critic_lr} + 1e-3}")

# done

# for((ITER=1; ITER<=3; ITER++))
# do
#     gamma=$(awk "BEGIN {print${gamma} + 0.01}")
#     lmbda=$(awk "BEGIN {print${lmbda} + 0.01}")
#     actor_lr=1e-4
#     critic_lr=1e-3
#     python gcc_trainer.py \
#         --gamma ${gamma} \
#         --lmbda ${lmbda} \
#         --actor_lr ${actor_lr} \
#         --critic_lr ${critic_lr} \
#         --eps ${eps} \
#         --n_hiddens ${n_hiddens}
# done
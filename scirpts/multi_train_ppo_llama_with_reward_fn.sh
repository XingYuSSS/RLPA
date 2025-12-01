# /openrlhf/examples/scripts/reward_func.py
# import torch

# def reward_func(queries, prompts, labels):
#     # queries is prompts + responses
#     # labels is answers
#     print(queries)
#     return torch.randn(len(queries))

set -x 

max_rounds=10
n_samples_per_prompt=4

micro_train_batch_size=4
train_batch_size=128
micro_rollout_batch_size=16
rollout_batch_size=256
round_batch_size=256

# micro_train_batch_size=4
# train_batch_size=96
# micro_rollout_batch_size=16
# rollout_batch_size=192

max_epochs=1
num_episodes=1

actor_learning_rate=5e-7
critic_learning_rate=9e-6

root_path=path_to_this_project

prompt_data_path=path_to_data

base_model_path=path_to_model
save_path=$root_path/saves/save_model_name

ray job submit --address="http://127.0.0.1:8265" \
   --no-wait \
   --runtime-env-json='{"working_dir": "path_to_workdir"}' \
   -- python3 -u -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 8 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 8 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 8 \
   --vllm_num_engines 8 \
   --vllm_tensor_parallel_size 1 \
   --pretrain $base_model_path \
   --remote_rm_url $root_path/scirpts/multi_turn_rfn.py \
   --save_path $save_path/final \
   --ckpt_path $save_path/ckpt \
   --save_hf_ckpt \
   --micro_train_batch_size $micro_train_batch_size \
   --train_batch_size $train_batch_size \
   --micro_rollout_batch_size $micro_rollout_batch_size \
   --rollout_batch_size $rollout_batch_size \
   --max_samples 10000000 \
   --max_epochs $max_epochs \
   --num_episodes $num_episodes \
   --prompt_max_len 4096 \
   --generate_max_len 4096 \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate $actor_learning_rate \
   --critic_learning_rate $critic_learning_rate \
   --init_kl_coef 0.01 \
   --prompt_data $prompt_data_path \
   --input_key input \
   --label_key profile \
   --apply_chat_template \
   --normalize_reward \
   --packing_samples \
   --flash_attn \
   --gradient_checkpointing \
   --max_rounds $max_rounds \
   --n_samples_per_prompt $n_samples_per_prompt \
   --use_multiturn true \
   --use_wandb True \
   --colocate_all_models \
   --vllm_gpu_memory_utilization 0.65 \
   --vllm_sync_backend nccl \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --save_steps 10 \
   --system_file  $root_path/scirpts/actor_sys.txt \
   --round_batch_size $round_batch_size


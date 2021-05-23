#!/bin/bash

. /benchmark.config

for seed in 122 343 234 542 484 294 595 947 101 484
do
  echo "Starting experiment for seed $seed"
  srun -n 4 --gres=gpu:4 --mpi=pmi2 python train_parallel.py --seed="$seed" --T-max="$T_max" --plan-steps="$plan_steps" \
  --mdp-epochs="$mdp_epochs" --policy-epochs="$policy_epochs" --critic-min-epochs="$critic_min_epochs" \
  --critic-max-epochs="$critic_max_epochs" --critic-loss-threshold="$critic_loss_threshold" \
  --network-h-units="$network_h_units" --network-h-layers="$network_h_layers" --n-explore-steps="$n_explore_steps" \
  --env="$env" --network-batchsize="$network_batchsize" --beta-threshold="$beta_threshold" --dataset-len="$dataset_len"\
  --network-epochs="$network_epochs" --llh-threshold="$llh_threshold" --n-last-llhs="$n_last_llhs" \
  --actor-h-units="$actor_h_units" --actor-h-layers="$actor_h_layers" --critic-batchsize="$critic_batchsize" \
  --critic-h-units="$critic_h_units" --critic-h-layers="$critic_h_layers" --actor-lr="$actor_lr" --critic-lr="$critic_lr"\
  --n-eval-episodes="$n_eval_episodes" --terminal-done="$terminal_done" --obs-scale="$obs_scale" --replay-capacity="$replay_capacity"\
  --tau="$tau" --id="gym_benchmark" --resume="$resume" --beta-stepsize="$beta_stepsize" --alpha="$alpha"
done

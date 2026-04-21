# Results

This folder contains representative training and evaluation plots for the RL-based adaptive cruise control experiments in this repo.

I wanted the repo to show not only the code, but also what the learned controller actually does, both during training and during rollout on a driving profile.

## What is here

Right now, I included two main FTP-75 SAC result sets:

- `ftp75_sac_ep200_per10/`
- `ftp75_sac_ep250_per50/`

These folders include a mix of:
- training/evaluation reward plots,
- actor and critic loss plots,
- rollout plots for speed tracking,
- spacing behavior,
- acceleration comparison,
- and cumulative energy consumption.

## Types of plots

Some of the result folders include plots like:

- `training_eval_combined_*.pdf`  
  training reward and evaluation reward across timesteps, usually averaged across multiple seeds

- `actor_loss_combined_*.pdf`  
  actor loss trend during training

- `critic_loss_combined_*.pdf`  
  critic loss trend during training

- `reference_vs_ego_speed.png`  
  comparison between lead/reference speed and ego vehicle speed

- `distance_difference.png`  
  distance gap over time between lead and ego vehicles

- `acceleration_comparison.png`  
  ego and lead acceleration comparison

- `cumulative_energy.png`  
  cumulative energy comparison between ego and lead vehicles

## Why I included these

The idea here was to show both sides of the project:

1. **training behavior**
   - whether the policy converges reasonably
   - how reward evolves
   - how actor/critic losses behave

2. **rollout behavior**
   - whether the ego vehicle tracks the reference well
   - whether spacing stays reasonable
   - how aggressive or smooth the response is
   - whether there is an energy benefit compared to the lead/reference case

## Not the full archive

I generated many more plots than what is included here, across different algorithms, packet-loss settings, episode lengths, and test cases.

I did not want the repo to turn into a storage dump, so I kept this folder focused on a smaller set of representative results.

## Folder naming

The result folders follow the same naming style as the model folders, for example:

`ftp75_sac_ep200_per10`

This makes it easy to connect:
- the trained model in `models/`
- the plots in `results/`
- and the experiment setting being shown

## In short

This folder is meant to give a quick but concrete view of how the controller behaved under a few representative settings, without overwhelming the repo with every single experiment output.

# Trained Models

This folder contains trained RL checkpoints for the adaptive cruise control experiments in this repo.

Right now, I included representative SAC models for a couple of FTP-75 setups so the repo has actual trained artifacts tied to the evaluation plots and experiment folders.

## What is here

At the moment, the main included examples are:

- `ftp75_sac_ep200_per10/`
  - SAC model
  - FTP-75 driving cycle
  - episode length: 200
  - packet error rate: 10%

- `ftp75_sac_ep250_per50/`
  - SAC model
  - FTP-75 driving cycle
  - episode length: 250
  - packet error rate: 50%

Each folder may contain a trained checkpoint such as:

- `sac_custom.zip`

## What these models are for

These checkpoints are meant to be used with the evaluation pipeline in this repo. They correspond to the same environment design, observation setup, action definition, and reward structure used during training.

In other words, they are not just random saved models — they are tied to the specific ACC + energy-aware EV setup used here.

## Bigger experiment set

These are only a small subset of the full experiments I ran.

Across the full set of experiments, I trained models using:
- DDPG
- SAC
- TD3

and varied:
- packet error rate: 0.1, 0.5, 0.9
- episode length: 100, 200, 250, 300

I did not want to dump every single checkpoint into the repo and make it messy, so for now I kept a representative subset here.

## Folder naming

I used folder names like:

`ftp75_sac_ep200_per10`

which means:
- `ftp75`: driving cycle
- `sac`: algorithm
- `ep200`: episode length
- `per10`: packet error rate = 10%

That way the model folders line up cleanly with the corresponding result folders.

## Note

If the code changes later, some older checkpoints may need small adjustments before they run directly again. But for the current setup, these folders are meant to match the code and the evaluation outputs included in the repo.

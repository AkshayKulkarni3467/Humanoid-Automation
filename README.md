# Humanoid Automation with SAC

## Overview:
This project trains the humanoid agent using SAC (Soft Actor Critic) algorithm.

## Overview of algorithm:
- Soft Actor Critic (SAC) is an algorithm that optimizes a stochastic policy in an off-policy way.
- It incorporates the clipped double-Q trick.
- A central feature of SAC is entropy regularization.


## After Training GIF:
![sac gif 1](https://github.com/AkshayKulkarni3467/Humanoid-Automation/assets/129979542/5c0a15b7-7bee-40e0-a37f-c6c9e2a236ac)

![sac gif 2](https://github.com/AkshayKulkarni3467/Humanoid-Automation/assets/129979542/d1aada33-6e77-4bc4-86bb-ba16cf808483)

## Loss and Reward curves:
- Actor Loss
  
![sac_policy_loss](https://github.com/AkshayKulkarni3467/Humanoid-Automation/assets/129979542/703cfc80-7df1-476e-a31e-25e4ca90ce29)

- Critic Loss

![sac_criticloss](https://github.com/AkshayKulkarni3467/Humanoid-Automation/assets/129979542/d7d6192b-f029-4e9b-828f-650fd9a94b78)

- Reward Curve

![sac_reward](https://github.com/AkshayKulkarni3467/Humanoid-Automation/assets/129979542/6cfafeb5-ac62-4582-9fab-be83d28d1f78)

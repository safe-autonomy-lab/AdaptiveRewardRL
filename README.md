# Adaptive Reward Design for Reinforcement Learning

**Paper Title:** Adaptive Reward Design for Reinforcement Learning

**Link to project page:** [https://github.com/RewardShaping/AdaptiveRewardShaping](https://github.com/RewardShaping/AdaptiveRewardShaping)

**Citation:** If you use this code or build upon it, please cite:

```bibtex
@inproceedings{Kwon2025AdaptiveReward,
  title     = {Adaptive Reward Design for Reinforcement Learning},
  author    = {Kwon, Minjae and ElSayed-Aly, Ingy and Feng, Lu},
  booktitle = {Proceedings of the Conference on Uncertainty in Artificial Intelligence (UAI)},
  year      = {2025},
}
```

**Contact:** For questions or collaborations, please contact:

Minjae Kwon - hbt9su@virginia.edu

## Overview

This repository implements our method for **Adaptive Reward Design for Reinforcement Learning**, addressing the common challenge of sparse rewards when using Linear Temporal Logic (LTL) to specify complex tasks. While LTL provides precision, sparse rewards (e.g., feedback only on full task completion) often make it difficult for RL agents to learn effectively.

Our approach overcomes this by:
* Introducing **LTL-derived reward functions** that provide denser feedback compared to typical goal-achieved reward functions (which might assign 1 for completion and 0 otherwise). Our denser feedback is based on the structure of the Deterministic Finite Automaton (DFA) derived from the LTL formula, where we consider each node in the DFA as a sub-task. With our reward design, this encourages agents to complete as much of a task as possible, not just the final goal.
* Developing an **adaptive reward shaping mechanism** that dynamically updates these LTL-derived reward functions during the learning process. This guides the agent more effectively based on its progress. While assigning partial rewards for solving sub-tasks (as illustrated in our toy example) can be beneficial, it also risks the policy converging to a sub-optimal solution that completes only early sub-tasks. Our adaptive reward design specifically addresses this issue of getting trapped in sub-optimal policies, a capability supported by theoretical guarantees (see Theorem 1 in our paper).

The key idea is to reward incremental progress and adapting the learning signals as the agent explores. This repository contains the code to reproduce the experiments and utilize the proposed methods. Experimental results on a range of benchmark RL environments demonstrate that our approach generally outperforms baselines, achieving earlier convergence to a better policy with higher expected return and task completion rate.

For more details, implementation, and experimental results, visit the main repository: [https://github.com/RewardShaping/AdaptiveRewardShaping](https://github.com/RewardShaping/AdaptiveRewardShaping). The paper is available on arXiv: [arXiv:2412.10917](https://arxiv.org/abs/2412.10917).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

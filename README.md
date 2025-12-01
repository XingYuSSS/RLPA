<div align="center">
  <h1>Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment</h1>
</div>

This repository contains code for the NeurIPS 2025 paper [Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment](https://arxiv.org/abs/2505.15456)

This repository is an extension and modification of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF), implementing multi-turn online reinforcement learning. Special thanks to the OpenRLHF team for their excellent work!


## Installation

We recommend a simple local installation workflow based on this repository.

```bash
git clone https://github.com/XingYuSSS/RLPA.git
cd RLPA

# (Optional) create and activate a conda environment
conda create -n rlpa python=3.10
conda activate rlpa

# Install this project and its dependencies
pip install -e .
```


## Quick Start

We provide a single script for multi-turn PPO training with a custom reward function:

- Script: `scirpts/multi_train_ppo_llama_with_reward_fn.sh`

### Key Parameters for Multi-turn Training

- `--max_rounds`: Maximum number of interaction rounds per multi-turn conversation during RL training.
- `--n_samples_per_prompt`: Number of response samples generated for each conversation history. The sample with the highest reward score will be selected to continue the conversation history.
- `--use_multiturn`: When set to `false`, the system behaves identically to the original OpenRLHF (single-turn training).
- `--remote_rm_url`: In multi-turn mode, the system reads the `user_func` function from the specified file to generate user replies for the next round of conversation.

### Training Steps

1. **Edit the script** `scirpts/multi_train_ppo_llama_with_reward_fn.sh`:
   - Set `root_path`, `prompt_data_path`, `base_model_path`, and `save_path` to the actual paths on your machine.
   - Adjust batch sizes, learning rates, `max_rounds`, `n_samples_per_prompt`, etc., according to your hardware and experimental setup.
   - Ensure `--remote_rm_url` in the script points to your reward function file (e.g., `scirpts/multi_turn_rfn.py`).
2. **Start Ray** (if needed) and run:

```bash
bash scirpts/multi_train_ppo_llama_with_reward_fn.sh
```

After these steps, the script will launch the multi-turn online RL training based on OpenRLHF and save checkpoints to the specified directories.


## Citation

```text
@article{zhao2025teachinglanguagemodelsevolve,
      title={Teaching Language Models to Evolve with Users: Dynamic Profile Modeling for Personalized Alignment}, 
      author={Weixiang Zhao and Xingyu Sui and Yulin Hu and Jiahe Guo and Haixiao Liu and Biye Li and Yanyan Zhao and Bing Qin and Ting Liu},
      year={2025},
      eprint={2505.15456},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.15456}, 
}
```

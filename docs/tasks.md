# CS 4650 Project Task Tracker

## Data Pipeline & Reward Model

| Task | Owner | Status | Blocks | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Download & preprocess New Yorker dataset | Yosie | Not started | Everything |  |
| Create train/val/test splits | Yosie | Not started | RM training, baselines |  |
| Format Bradley-Terry preference pairs | Yosie | Not started | RM training | 3-sigma difference pairs (from paper) |
| Set up Qwen3-VL \+ LoRA config | Yosie | Not started | RM training | 2B size is good enough to start |
| Implement Bradley-Terry RM training loop | Yosie | Not started | RM training | trl RewardTrainer |
| Train reward model | Yosie | Not started | GRPO policy training |  |
| Evaluate RM human agreement (Exp 1\) | Yosie | Not started | Final report | check it matches held-out human ranking in dataset |

## Infrastructure & Policy Model

| Task | Owner | Status | Blocks | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Set up shared training env (GPUs, W\&B) | David | Done | Training runs |  |
| Implement GRPO pipeline (stub reward) | David | In progress | Policy training | Gemma 4 E2B-it + LoRA via Unsloth + TRL GRPOTrainer; scripts/train_grpo.py, scripts/train_grpo.slurm, scripts/reward_stub.py |
| Design CoT prompt template | David | Not started | Policy training |  |
| Train policy model with GRPO | David | Not started | Exps 2 & 3 | Waiting on: trained RM |
| Generate policy captions | David | Not started | Exps 2 & 3 |  |

## Baselines & Evaluation

| Task | Owner | Status | Blocks | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Qwen3-VL zero-shot baseline | Erika | Not started | Final results | Waiting on: data splits |
| Qwen3-VL few-shot baseline | Erika | Not started | Final results |  |
| Qwen3-VL SFT baseline | Erika | Not started | Final results |  |
| Curate OOD image dataset | Erika | Not started | Exp 3 | No dependencies, start anytime |
| Implement LLM eval (Zhang et al.) | Erika | Not started | Exps 2 & 3 |  |
| Design human eval survey | Erika | Not started | Exp 2 |  |
| Run human eval on policy captions (Exp 2\) | Erika | Not started | Final report | Waiting on: policy outputs |
| Run OOD eval (Exp 3\) | Erika | Not started | Final report | Waiting on: policy outputs |
| Compile results tables | Erika | Not started | Final report |  |

## Writeup

| Task | Owner | Status | Blocks | Notes |
| :---- | :---- | :---- | :---- | :---- |
| Write final report | All | Not started | — | Each person writes their section |

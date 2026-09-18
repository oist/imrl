# IMRL: Intrinsically Motivated Reinforcement Learning

Minimal repository for running IMRL experiments on:

- MiniGrid FourRooms TwoGoals RandKey (3x3 view)
- Craftax Classic Symbolic 32x32 (9x9 view)

## Supported Python versions

3.9 to 3.12

## Installation

1. Clone the repository and enter it.
2. Create and activate a Python virtual environment.
3. Install dependencies.

```bash
git clone <this-imrl-repo-url>
cd imrl

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Optional (for Weights & Biases logging):

```bash
wandb login
```

## Example Runs

### 1) MiniGrid run

```bash
python imrl.py --env MiniGrid-FourRooms-TwoGoals-RandKey-ViewSize-3x3-v0 --reward-type combined --timesteps 1e6 --int-rew-coef 1 --novelty-weight 1e-3 --surprise-weight 1e-3 --empowerment-weight 1e-4 --save-best-policy --device auto --record-episodes --seed 23 --wandb-project minigrid_test --checkpoint-dir tmp_test_run --checkpoint-freq 5
```

### 2) Craftax Classic Symbolic 32x32 run

```bash
python imrl.py --env Craftax-Classic-Symbolic-32x32-v1 --reward-type combined --timesteps 1e4 --int-rew-coef 1 --novelty-weight 1e-3 --surprise-weight 1e-3 --empowerment-weight 1e-2 --save-best-policy --device auto --record-episodes --seed 23 --wandb-project craftax_test --checkpoint-dir tmp_test_run --checkpoint-freq 5
```

### 3) Evolutionary optimization of intrinsic reward weights

Runs a generational genetic algorithm that searches for good novelty/surprise/empowerment
weight mixtures by training and evaluating many genomes in parallel (no SLURM required).

```bash
python -m evolutionary_optim.local_evolutionary_coordinator --env-id MiniGrid-FourRooms-TwoGoals-RandKey-ViewSize-3x3-v0 --population-size 20 --max-generations 20 --n-seeds 3 --n-timesteps 1000000 --n-workers 4 --n-envs 128 --n-gpus 1 --results-dir local_evo_results
```

This also works on a Craftax environment (weight bounds are chosen automatically per environment):

```bash
python -m evolutionary_optim.local_evolutionary_coordinator --env-id Craftax-Classic-Symbolic-32x32-v1 --population-size 20 --max-generations 20 --n-seeds 3 --n-timesteps 1000000 --n-workers 4 --n-envs 128 --n-gpus 1 --results-dir local_evo_results_craftax
```

Add `--resume` to continue from `<results-dir>/checkpoint.json` if a run was interrupted.

## Notes

- `--checkpoint-dir tmp_test_run` will create checkpoints in the repository folder.
- `--record-episodes` enables episode recording during training.
- `--device auto` selects GPU when available, otherwise CPU.
- `requirements.txt` installs `jax` with the CUDA 12 extra on Linux, and plain (CPU) `jax` on
  other platforms (e.g. local macOS development), since CUDA wheels only exist for Linux.

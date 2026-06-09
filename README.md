<p align="center">
  <img src="./docs/roleagent.png" alt="Role-Agent" width="58%">
</p>

<h1 align="center">Role-Agent: Bootstrapping LLM Agents via Dual-Role Evolution</h1>

<p align="center">
  <b>World-In-Agent reasoning + Agent-In-World curriculum for interactive LLM agent training</b>
</p>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="./docs/role_agent_alignment.md"><img src="https://img.shields.io/badge/docs-Role--Agent-green?style=flat-square" alt="Docs"></a>
  <a href="./examples/role_agent_trainer/"><img src="https://img.shields.io/badge/examples-ALFWorld%20%7C%20WebShop%20%7C%20Search--R1-orange?style=flat-square" alt="Examples"></a>
</p>

---

## Overview

Role-Agent extends the [`verl-agent`](https://github.com/langfengQ/verl-agent)
and [`veRL`](https://github.com/volcengine/verl) training stack with
**dual-role evolution** for LLM agents. The project keeps the scalable
multi-turn rollout and RL infrastructure from the upstream stack, while adding
two Role-Agent components that make the agent learn both as an internal world
model and as an actor shaped by past world failures.

| Component | What It Adds | Main Entry Points |
| --- | --- | --- |
| **World-In-Agent (WIA)** | Predict next environment feedback with `<predict_next>` and shape step rewards by prediction-observation similarity. | `role_agent/wia_utils.py`, `agent_system/multi_turn_rollout/rollout_loop.py` |
| **Agent-In-World (AIW)** | Track failed episodes as lightweight failure fingerprints and upweight similar tasks during training. | `role_agent/aiw_curriculum.py`, `verl/trainer/ppo/ray_trainer.py` |
| **Training Integration** | Toggle Role-Agent behavior through `algorithm.role_agent.*` without replacing the existing PPO/GiGPO pipeline. | `verl/trainer/config/ppo_trainer.yaml` |
| **Launch Recipes** | Ready-to-run scripts for ALFWorld, WebShop, Search-R1, and WebShop + GiGPO. | `examples/role_agent_trainer/` |

For detailed design notes, implementation alignment, and known tradeoffs, see
[`docs/role_agent_alignment.md`](./docs/role_agent_alignment.md).

## Highlights

- **Dual-role training signal:** combines next-state prediction quality with
  failure-aware task resampling.
- **Minimal integration surface:** WIA/AIW are optional Hydra flags layered on
  top of the existing multi-turn rollout loop.
- **Long-horizon friendly:** inherits `verl-agent`'s step-independent rollout
  design, avoiding full-history concatenation at every turn.
- **Practical training recipes:** includes concrete scripts for text-based
  environments and search/tool-use settings.

## Quick Start

Install dependencies following the upstream `verl-agent` / `veRL` environment
requirements, then run a Role-Agent recipe:

```bash
cd /path/to/roleagent
bash examples/role_agent_trainer/run_webshop.sh
```

Enable Role-Agent behavior with Hydra flags:

```bash
algorithm.role_agent.enable_wia=true \
algorithm.role_agent.enable_aiw=true
```

Common configuration keys:

| Key | Purpose |
| --- | --- |
| `algorithm.role_agent.text_match_max_chars` | Cap text length for WIA/AIW similarity scoring. |
| `algorithm.role_agent.aiw_top_k` | Number of similar failed tasks to upweight. |
| `algorithm.role_agent.aiw_boost` | Cross-task AIW sampling boost. |
| `algorithm.role_agent.aiw_self_boost` | Failed-task self replay boost. |
| `algorithm.role_agent.aiw_similarity_thresh` | Optional similarity gate for cross-task boosts. |
| `algorithm.role_agent.aiw_max_history` | Maximum retained failure fingerprints. |

When AIW is enabled, set `data.dataloader_num_workers=0` so the mutable
weighted sampler remains well-defined.

## Example Recipes

| Script | Environment | Algorithm |
| --- | --- | --- |
| `examples/role_agent_trainer/run_alfworld.sh` | ALFWorld | PPO / GAE |
| `examples/role_agent_trainer/run_webshop.sh` | WebShop | PPO / GAE |
| `examples/role_agent_trainer/run_webshop_gigpo.sh` | WebShop | GiGPO |
| `examples/role_agent_trainer/run_search.sh` | Search-R1 | GiGPO |

Data-root overrides are documented in
[`examples/role_agent_trainer/README.md`](./examples/role_agent_trainer/README.md).

## Repository Map

```text
role_agent/                         # WIA scoring, AIW curriculum, prompt utilities
agent_system/multi_turn_rollout/    # Multi-turn rollout loop with Role-Agent hooks
verl/trainer/ppo/ray_trainer.py     # PPO/GiGPO trainer integration
examples/role_agent_trainer/        # Role-Agent launch scripts
docs/role_agent_alignment.md        # Detailed design and implementation notes
```

## Upstream Base

This repository is derived from `verl-agent`, which provides scalable
multi-turn interaction, environment wrappers, and RL algorithms including PPO,
GRPO, DAPO, RLOO, and GiGPO. Role-Agent focuses on the WIA/AIW training
additions on top of that base.

Please also acknowledge the upstream projects when using this code:

- [`verl-agent`](https://github.com/langfengQ/verl-agent)
- [`veRL`](https://github.com/volcengine/verl)

## License

This repository follows the upstream Apache-2.0 license. See
[`LICENSE`](./LICENSE).

# thirdparty

The upstream code for every paper vlalab implements, wraps or runs. Each is a git submodule, pinned to the
commit it was checked against, so you can read the original next to the component that implements it here.
A plain clone fetches none of it.

```bash
git submodule update --init --depth 1 thirdparty/openpi     # one
git submodule update --init --depth 1                        # all 31, about 2.5 GB
```

LeRobot, robotics_transformer and Isaac-GR00T keep large files in Git LFS. Prefix the command with
`GIT_LFS_SKIP_SMUDGE=1` to fetch only their code.

**use** says how vlalab uses the code:

* **runs**: vlalab runs this exact code. The pin is the version that runs:
  * the pip version in `uv.lock` for LeRobot, gym-pusht and gym-aloha;
  * the commit `scripts/real2sim/setup_*.sh` checks out for UMI, SAM 3D Objects, Steel and YCB (the scripts read it
    from this folder);
  * the commit those repos pin in turn: Newton from Steel's `examples/pyproject.toml`, MoGe from SAM 3D's
    `requirements.txt`, newton-assets from Newton.
* **via 🤗**: the model runs through transformers or LeRobot, from the weights listed under
  [Hugging Face models](../README.md#hugging-face-models). The repo is the authors' reference code.
* **reference**: vlalab re-implements the method (the component is named); the original is here to compare against.

Libraries such as torch, transformers, peft and torchao are pinned by `uv.lock`, not here.

## 01 · VLM

| submodule | pin | paper | in vlalab | use | licence |
|---|---|---|---|---|---|
| [smollm](https://github.com/huggingface/smollm) | `438be18` | [SmolVLM2](https://arxiv.org/abs/2504.05299) | `backbone/hf_vlm`, `vlm/hf_pretrained`, `vlm/er_finetune` | via 🤗 | Apache-2.0 |
| [robotics_transformer](https://github.com/google-research/robotics_transformer) | `4569641` | [RT-1](https://arxiv.org/abs/2212.06817) | `backbone/tiny_cnn` (FiLM CNN + tokens) | reference | Apache-2.0 |

## 02 · Data

| submodule | pin | paper | in vlalab | use | licence |
|---|---|---|---|---|---|
| [lerobot](https://github.com/huggingface/lerobot) | `v0.7.0` | LeRobot | `data/lerobot_hub`, `vla/lerobot_train`, the LeRobot policy runner in eval, `driver/lerobot_robot`; SmolVLA, ACT, π0 / π0.5, GR00T, X-VLA and HIL-SERL run through it | runs | Apache-2.0 |
| [open_x_embodiment](https://github.com/google-deepmind/open_x_embodiment) | `9eeb68b` | [Open X-Embodiment](https://arxiv.org/abs/2310.08864) | `data/mixture` | reference | Apache-2.0 |

## 03 · VLA and 04 · Post-training

| submodule | pin | paper | in vlalab | use | licence |
|---|---|---|---|---|---|
| [openpi](https://github.com/Physical-Intelligence/openpi) | `215abfb` | [π0](https://arxiv.org/abs/2410.24164), π0.5 | `action_head/flow_matching`, `vla/train`, `presets/vla/pi0_style` | reference | Apache-2.0 |
| [diffusion_policy](https://github.com/real-stanford/diffusion_policy) | `5ba07ac` | [Diffusion Policy](https://arxiv.org/abs/2303.04137) | `action_head/diffusion`; the PushT task | reference | MIT |
| [octo](https://github.com/octo-models/octo) | `241fb35` | [Octo](https://arxiv.org/abs/2405.12213) | `action_head/diffusion` (Octo's diffusion head) | reference | MIT |
| [openvla](https://github.com/openvla/openvla) | `c8f03f4` | [OpenVLA](https://arxiv.org/abs/2406.09246) | `action_head/discrete_bins` (256-bin action tokens, as in RT-2); `post_train/finetune` `method: lora` | reference | MIT |
| [openvla-oft](https://github.com/moojink/openvla-oft) | `e4287e9` | [OpenVLA-OFT](https://arxiv.org/abs/2502.19645) | `action_head/l1_regression`, `presets/vla/openvla_oft`, `post_train/finetune` | reference | MIT |
| [act](https://github.com/tonyzhaozh/act) | `742c753` | [ACT](https://arxiv.org/abs/2304.13705) | `chunker/sync`, `chunker/temporal_ensemble`; the ACT policy runs through LeRobot | reference | MIT |
| [Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) | `n1.5-release` | [GR00T N1](https://arxiv.org/abs/2503.14734), N1.5 | `post_train/finetune` `method: frozen_backbone` | reference | Apache-2.0 |
| [X-VLA](https://github.com/2toinf/X-VLA) | `6c4ecfb` | [X-VLA](https://arxiv.org/abs/2510.10274) | `vla/lerobot_train` `policy_path: lerobot/xvla-base` | via 🤗 | Apache-2.0 |
| [fourier-feature-networks](https://github.com/tancik/fourier-feature-networks) | `9c110c3` | [Fourier features](https://arxiv.org/abs/2006.10739) | `obs_encoder/fourier` | reference | MIT |

## 05 · RL

| submodule | pin | paper | in vlalab | use | licence |
|---|---|---|---|---|---|
| [dsrl](https://github.com/ajwagen/dsrl) | `8d21b9c` | [DSRL](https://arxiv.org/abs/2506.15799) | `rl/dsrl` | reference | none given |
| [hil-serl](https://github.com/rail-berkeley/hil-serl) | `c32939b` | [HIL-SERL](https://arxiv.org/abs/2410.21845) | LeRobot's actor / learner (`lerobot/rl`) | via 🤗 | Apache-2.0 |

## 06 · Evaluation

| submodule | pin | paper | in vlalab | use | licence |
|---|---|---|---|---|---|
| [gym-pusht](https://github.com/huggingface/gym-pusht) | `v0.1.8` | PushT, from Diffusion Policy | `env/gym` with `configs/env/pusht.yaml` | runs | Apache-2.0 |
| [gym-aloha](https://github.com/huggingface/gym-aloha) | `v0.1.5` | ALOHA sim, from ACT | `env/gym` with `configs/env/aloha_transfer_cube.yaml` | runs | Apache-2.0 |
| [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) | `8f1084e` | [LIBERO](https://arxiv.org/abs/2306.03310) | `env/gym`, `eval/sim_rollout` (seeded sim eval) | reference | MIT |
| [SimplerEnv](https://github.com/simpler-env/SimplerEnv) | `eddb569` | [SimplerEnv](https://arxiv.org/abs/2405.05941) | `eval/sim_rollout` | reference | MIT |

## 08 · Deploy

| submodule | pin | paper | in vlalab | use | licence |
|---|---|---|---|---|---|
| [real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) | `9296f31` | [Real-time chunking](https://arxiv.org/abs/2506.07339) | `chunker/rtc_blend` (an output-space approximation), `deploy/loop` | reference | MIT |

## Real-to-sim (`umi_real2sim`)

| submodule | pin | paper | in vlalab | use | licence |
|---|---|---|---|---|---|
| [universal_manipulation_interface](https://github.com/real-stanford/universal_manipulation_interface) | `d095ba9` | [UMI](https://arxiv.org/abs/2402.10329) | `data/umi_demo` runs its SLAM pipeline | runs | MIT |
| [sam3](https://github.com/facebookresearch/sam3) | `31cb7e6` | [SAM 3](https://arxiv.org/abs/2511.16719) | `data/scene_masks` | via 🤗 | SAM License |
| [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | `856dde2` | [Grounding DINO](https://arxiv.org/abs/2303.05499) | `data/scene_masks` `backend: gdino_sam2` (detector) | via 🤗 | Apache-2.0 |
| [sam2](https://github.com/facebookresearch/sam2) | `2b90b9f` | [SAM 2](https://arxiv.org/abs/2408.00714) | `data/scene_masks` `backend: gdino_sam2` (segmenter) | via 🤗 | Apache-2.0 |
| [sam-3d-objects](https://github.com/facebookresearch/sam-3d-objects) | `f91db41` | [SAM 3D Objects](https://arxiv.org/abs/2511.16624) | `data/sam3d_scene` | runs | SAM License |
| [MoGe](https://github.com/microsoft/MoGe) | `a8c3734` | [MoGe](https://arxiv.org/abs/2410.19115) | `data/sam3d_scene`: the point map SAM 3D lays objects on, and `method: moge` | runs | MIT |
| [CLIP](https://github.com/openai/CLIP) | `d05afc4` | [CLIP](https://arxiv.org/abs/2103.00020) | `data/physics_match` (the matching classifier) | via 🤗 | MIT |
| [ycb_dataset](https://github.com/elpis-lab/ycb_dataset) | `9e8c648` | [YCB](https://arxiv.org/abs/1502.03143) | `data/physics_match`: measured masses and meshes | runs | MIT |
| [newton](https://github.com/skild-ai/newton) | `f6fc82a` | Newton, Skild AI's fork with the Steel work | `data/steel_scene`: physics, IK, USD export | runs | Apache-2.0 |
| [newton-assets](https://github.com/newton-physics/newton-assets) | `f8fb7ab` | Franka FR3 + hand URDF | `data/steel_scene`: the robot | runs | Apache-2.0 |
| [steel](https://github.com/skild-ai/steel) | `v0.2.0` | [Steel](https://www.skild.ai/blogs/steel) | `data/steel_scene`: `SolverSteel` | runs | Apache-2.0 |

## Changing a pin

```bash
git -C thirdparty/steel fetch --depth 1 origin <commit> && git -C thirdparty/steel checkout <commit>
git add thirdparty/steel
```

For a **runs** submodule, move what runs too. For LeRobot and the gym sims, change the version in
`pyproject.toml` and run `uv lock`. For the real-to-sim repos, re-run their `scripts/real2sim/setup_*.sh`. Each
setup script also takes a one-off override, for example `STEEL_REV=main`.

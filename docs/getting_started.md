# Getting started

[← README](../README.md)

- [Install](#install)
- [A GPU VM on GCP](#a-gpu-vm-on-gcp)
- [The pipelines](#the-pipelines)
- [Check every pipeline](#check-every-pipeline)
- [Where next](#where-next)

## Install

Linux with an NVIDIA GPU, or a cloud VM:

```bash
git clone https://github.com/aPR0T0/robot-learning && cd robot-learning
./setup.sh              # apt deps (EGL, ffmpeg), uv, .venv with CUDA torch + LeRobot sims, `vlab doctor`
source .venv/bin/activate
vlab run reach_smoke    # all eight stages, ~3 minutes on one L4
```

`./setup.sh --minimal` skips LeRobot [[1](#ref-1)]. CPU-only machines work too (`reach_smoke` takes ~23 min
on 8 vCPUs).

`setup.sh` also installs NVIDIA's EGL library when the driver image lacks it (GCP's accelerator images
do). Without it MuJoCo [[2](#ref-2)] silently renders on the CPU and the ALOHA sim is ~17× slower. `vlab
doctor` says which renderer you got.

## A GPU VM on GCP

```bash
scripts/gcp_vm.sh create        # 1x L4 (g2-standard-8), Ubuntu 24.04 + NVIDIA 580 driver image
scripts/gcp_vm.sh setup         # copy this checkout to the VM and run ./setup.sh there
scripts/gcp_vm.sh run reach_smoke
scripts/gcp_vm.sh pull          # copy run metrics, reports and logs back (no checkpoints)
scripts/gcp_vm.sh stop          # when done
```

Set `ZONE=...` if a zone is out of L4s. The other settings (`PROJECT`, `VM`, `MACHINE`, `GPU`,
`DISK_GB`, `SPOT=1`) and subcommands (`push`, `pull-all`, `ssh`, `start`, `delete`) are listed at the
top of [scripts/gcp_vm.sh](../scripts/gcp_vm.sh).

## The pipelines

| pipeline | what it shows | time (1× L4) |
|---|---|---|
| `reach_smoke` | all 8 stages on the built-in Lang-Reach 2D sim with a tiny from-scratch backbone | 3.5 min |
| `reach_full` | same, with **SmolVLM2-256M** [[3](#ref-3)]: pointing fine-tune → SmolVLA-style [[4](#ref-4)] flow policy | 10.5 min |
| `reach_heads.py` | a **Python** pipeline: one dataset, four action heads (π0 flow [[5](#ref-5)], diffusion [[6](#ref-6)], OFT L1 [[7](#ref-7)], RT-2 bins [[8](#ref-8)]) | 4.5 min |
| `pusht_lerobot` | LeRobot PushT data + sim; published Diffusion Policy baseline vs the native policy | 10 min |
| `aloha_lerobot` | bimanual ALOHA sim [[9](#ref-9)]: published ACT checkpoint; SmolVLA fine-tune via `lerobot-train` | 14 min |
| `mock_hardware` | a robot that does not exist yet (ALOHA + wrist F/T + tactile) through the 50 Hz real-time loop | 16 s |
| `so100_lerobot` | SO-100 arm on real data (SmolVLA's pick-and-place set): SmolVLA-style policy, held-out open-loop eval, 30 Hz loop on a mock arm | ~25 min |

They live in [configs/pipeline/](../configs/pipeline/). The numbers each one produced on the last
verified run are in [Results](results.md), and on the [node map](https://apr0t0.github.io/robot-learning/)
(click a pipeline node).

## Check every pipeline

`scripts/verify_all.sh [pipelines...]` runs pipelines back to back and prints a pass/fail table. On a
VM, `scripts/gcp_vm.sh pull` brings the metrics back. `vlab doctor` checks GPU, CUDA, EGL and the
installed extras.

## Where next

- [Running pipelines and single stages](running.md): the run folder, caching, `--only` / `--from`, and
  every `vlab` command.
- The eight stage pages, starting with [01 VLM](stages/01_vlm.md).
- [Editing the YAML configs](configuring_yaml.md) and [adding a component](adding_components.md).

## References

1. <a id="ref-1"></a>R. Cadene, S. Alibert, F. Capuano, et al. *LeRobot: An Open-Source Library for End-to-End Robot Learning.* ICLR 2026. [arXiv:2602.22818](https://arxiv.org/abs/2602.22818)
2. <a id="ref-2"></a>E. Todorov, T. Erez, and Y. Tassa. *MuJoCo: A physics engine for model-based control.* IROS 2012. [doi:10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109)
3. <a id="ref-3"></a>A. Marafioti, O. Zohar, M. Farré, et al. *SmolVLM: Redefining small and efficient multimodal models.* arXiv preprint, 2025. [arXiv:2504.05299](https://arxiv.org/abs/2504.05299)
4. <a id="ref-4"></a>M. Shukor, D. Aubakirova, F. Capuano, et al. *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.* arXiv preprint, 2025. [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)
5. <a id="ref-5"></a>K. Black, N. Brown, D. Driess, et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* RSS 2025. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
6. <a id="ref-6"></a>C. Chi, Z. Xu, S. Feng, et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
7. <a id="ref-7"></a>M. J. Kim, C. Finn, and P. Liang. *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success.* RSS 2025. [arXiv:2502.19645](https://arxiv.org/abs/2502.19645)
8. <a id="ref-8"></a>A. Brohan, N. Brown, J. Carbajal, et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.* CoRL 2023. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
9. <a id="ref-9"></a>T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)

<details>
<summary>BibTeX for this page</summary>

```bibtex
@inproceedings{cadene2026lerobot,
  title         = {{LeRobot: An Open-Source Library for End-to-End Robot Learning}},
  author        = {Remi Cadene and Simon Alibert and Francesco Capuano and Michel Aractingi and Adil Zouitine and Pepijn Kooijmans and Jade Choghari and Martino Russi and Caroline Pascal and Steven Palma and Mustafa Shukor and Jess Moss and Alexander Soare and Dana Aubakirova and Quentin Lhoest and Quentin Gallouédec and Thomas Wolf},
  booktitle     = {International Conference on Learning Representations (ICLR)},
  year          = {2026},
  eprint        = {2602.22818},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2602.22818}
}

@inproceedings{todorov2012mujoco,
  title         = {{MuJoCo: A physics engine for model-based control}},
  author        = {Emanuel Todorov and Tom Erez and Yuval Tassa},
  booktitle     = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages         = {5026--5033},
  year          = {2012},
  doi           = {10.1109/IROS.2012.6386109}
}

@article{marafioti2025smolvlm,
  title         = {{SmolVLM: Redefining small and efficient multimodal models}},
  author        = {Andrés Marafioti and Orr Zohar and Miquel Farré and Merve Noyan and Elie Bakouch and Pedro Cuenca and Cyril Zakka and Ben Allal, Loubna and Anton Lozhkov and Nouamane Tazi and Vaibhav Srivastav and Joshua Lochner and Hugo Larcher and Mathieu Morlon and Lewis Tunstall and Leandro von Werra and Thomas Wolf},
  journal       = {arXiv preprint arXiv:2504.05299},
  year          = {2025},
  eprint        = {2504.05299},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2504.05299}
}

@article{shukor2025smolvla,
  title         = {{SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics}},
  author        = {Mustafa Shukor and Dana Aubakirova and Francesco Capuano and Pepijn Kooijmans and Steven Palma and Adil Zouitine and Michel Aractingi and Caroline Pascal and Martino Russi and Andres Marafioti and Simon Alibert and Matthieu Cord and Thomas Wolf and Remi Cadene},
  journal       = {arXiv preprint arXiv:2506.01844},
  year          = {2025},
  eprint        = {2506.01844},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2506.01844}
}

@inproceedings{black2025pi0,
  title         = {{{$\pi_0$}: A Vision-Language-Action Flow Model for General Robot Control}},
  author        = {Kevin Black and Noah Brown and Danny Driess and Adnan Esmail and Michael Equi and Chelsea Finn and Niccolo Fusai and Lachy Groom and Karol Hausman and Brian Ichter and Szymon Jakubczak and Tim Jones and Liyiming Ke and Sergey Levine and Adrian Li-Bell and Mohith Mothukuri and Suraj Nair and Karl Pertsch and Lucy Xiaoyang Shi and James Tanner and Quan Vuong and Anna Walling and Haohuan Wang and Ury Zhilinsky},
  booktitle     = {Robotics: Science and Systems (RSS)},
  year          = {2025},
  eprint        = {2410.24164},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2410.24164}
}

@inproceedings{chi2023diffusion,
  title         = {{Diffusion Policy: Visuomotor Policy Learning via Action Diffusion}},
  author        = {Cheng Chi and Zhenjia Xu and Siyuan Feng and Eric Cousineau and Yilun Du and Benjamin Burchfiel and Russ Tedrake and Shuran Song},
  booktitle     = {Robotics: Science and Systems (RSS)},
  year          = {2023},
  eprint        = {2303.04137},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2303.04137}
}

@inproceedings{kim2025oft,
  title         = {{Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success}},
  author        = {Moo Jin Kim and Chelsea Finn and Percy Liang},
  booktitle     = {Robotics: Science and Systems (RSS)},
  year          = {2025},
  eprint        = {2502.19645},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2502.19645}
}

@inproceedings{brohan2023rt2,
  title         = {{RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control}},
  author        = {Anthony Brohan and Noah Brown and Justice Carbajal and Yevgen Chebotar and Xi Chen and Krzysztof Choromanski and Tianli Ding and Danny Driess and Avinava Dubey and Chelsea Finn and others},
  booktitle     = {Conference on Robot Learning (CoRL)},
  year          = {2023},
  eprint        = {2307.15818},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2307.15818}
}

@inproceedings{zhao2023act,
  title         = {{Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware}},
  author        = {Tony Z. Zhao and Vikash Kumar and Sergey Levine and Chelsea Finn},
  booktitle     = {Robotics: Science and Systems (RSS)},
  year          = {2023},
  eprint        = {2304.13705},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2304.13705}
}
```

</details>

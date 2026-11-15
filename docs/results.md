# Results

[← README](../README.md)

Verified on 1× NVIDIA L4, 2026-11-14. What each pipeline does and how long it takes is in
[Getting started](getting_started.md#the-pipelines). Measured numbers from the last verified run are
also on the [node map](https://apr0t0.github.io/robot-learning/) (click a pipeline node).

Success rates come with 95% Wilson score intervals [[1](#ref-1)]; n is the number of seeded sim episodes.

## A web-pretrained VLM vs a from-scratch backbone

| | from-scratch backbone (`reach_smoke`) | SmolVLM2-256M [[2](#ref-2)] (`reach_full`) |
|---|---|---|
| pretrained policy, colours seen in training (n=64) | 98% (92–100) | 97% (89–99) |
| **zero-shot on an unseen colour** (n=64) | **2%** (0–8) | **34%** (24–47) |
| after post-training on 24 demos | 28% | 95% |
| after RL (filtered self-imitation [[3](#ref-3), [4](#ref-4)]) | 80% | 97% |
| final eval (n=200) | 78% (72–83) | 93.5% (89–96) |
| edge build (fewer flow steps), eval (n=200) | 79%, 1.8× faster | 98%, 1.2× faster |
| pointing error of the VLM before → after its fine-tune | – | 0.79 → 0.08 |

The web-pretrained VLM knows "purple" without ever seeing it in robot data, which is the RT-2 effect
[[5](#ref-5)] from the blog. The stages behind each row: [01 VLM](stages/01_vlm.md),
[03 VLA](stages/03_vla.md), [04 post-training](stages/04_post_train.md), [05 RL](stages/05_rl.md),
[06 evaluation](stages/06_eval.md), [07 edge](stages/07_edge.md).

## Four action heads on the same data

| action head (`reach_heads.py`, same data, 1500 steps, n=64) | success |
|---|---|
| flow matching (π0 [[6](#ref-6)]) | 100% (94–100) |
| diffusion, x0-prediction [[7](#ref-7), [8](#ref-8)] | 98% (92–100) |
| L1 regression (OpenVLA-OFT [[9](#ref-9)] / ACT [[10](#ref-10)]) | 97% (89–99) |
| 256-bin tokens (RT-2 [[5](#ref-5)] / OpenVLA [[11](#ref-11)]) | 20% (12–32) |

The bins head fits its training data (77% exact-bin accuracy) but compounds errors in closed loop. An
epsilon-prediction diffusion head reached only 28%, and 22% after 6000 steps, so x0-prediction is the
default. The heads are described in [03 VLA](stages/03_vla.md).

## Published checkpoints through this repo's eval node

| published checkpoints run through this repo's eval node | here | published |
|---|---|---|
| Diffusion Policy [[7](#ref-7)], PushT (n=50) | 74% (60–84) | 65.4% |
| ACT [[10](#ref-10)], ALOHA transfer cube (n=20) | 80% (58–92) | 83% |

Both checkpoints are LeRobot's [[12](#ref-12)]. SmolVLA [[13](#ref-13)] fine-tuned from `lerobot/smolvla_base` for
2000 steps (10.5 min) reaches 30% (15–52) on ALOHA transfer cube. The native tiny flow policy trained
5000 steps on `lerobot/pusht` reaches 10% (4–21). Both are budget-limited, not broken. The mock 50 Hz
loop held 20.0 ms ticks (0.02 ms jitter) with 44 ms inference, because chunked actions keep the queue
full ([08 Deploy](stages/08_deploy.md)).

The SO-100 real-data results (open-loop error on held-out episodes, 30 Hz loop timing) are in
[New hardware](hardware.md#example-an-so-100-arm-from-real-demos-to-the-30-hz-loop).

## References

1. <a id="ref-1"></a>E. B. Wilson. *Probable Inference, the Law of Succession, and Statistical Inference.* *Journal of the American Statistical Association* 22(158):209–212, 1927. [doi:10.1080/01621459.1927.10502953](https://doi.org/10.1080/01621459.1927.10502953)
2. <a id="ref-2"></a>A. Marafioti, O. Zohar, M. Farré, et al. *SmolVLM: Redefining small and efficient multimodal models.* arXiv preprint, 2025. [arXiv:2504.05299](https://arxiv.org/abs/2504.05299)
3. <a id="ref-3"></a>C. Gulcehre, T. Le Paine, S. Srinivasan, et al. *Reinforced Self-Training (ReST) for Language Modeling.* arXiv preprint, 2023. [arXiv:2308.08998](https://arxiv.org/abs/2308.08998)
4. <a id="ref-4"></a>J. Peters and S. Schaal. *Reinforcement Learning by Reward-Weighted Regression for Operational Space Control.* ICML 2007. [doi:10.1145/1273496.1273590](https://doi.org/10.1145/1273496.1273590)
5. <a id="ref-5"></a>A. Brohan, N. Brown, J. Carbajal, et al. *RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control.* CoRL 2023. [arXiv:2307.15818](https://arxiv.org/abs/2307.15818)
6. <a id="ref-6"></a>K. Black, N. Brown, D. Driess, et al. *π0: A Vision-Language-Action Flow Model for General Robot Control.* RSS 2025. [arXiv:2410.24164](https://arxiv.org/abs/2410.24164)
7. <a id="ref-7"></a>C. Chi, Z. Xu, S. Feng, et al. *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.* RSS 2023. [arXiv:2303.04137](https://arxiv.org/abs/2303.04137)
8. <a id="ref-8"></a>J. Ho, A. Jain, and P. Abbeel. *Denoising Diffusion Probabilistic Models.* NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
9. <a id="ref-9"></a>M. J. Kim, C. Finn, and P. Liang. *Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success.* RSS 2025. [arXiv:2502.19645](https://arxiv.org/abs/2502.19645)
10. <a id="ref-10"></a>T. Z. Zhao, V. Kumar, S. Levine, and C. Finn. *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.* RSS 2023. [arXiv:2304.13705](https://arxiv.org/abs/2304.13705)
11. <a id="ref-11"></a>M. J. Kim, K. Pertsch, S. Karamcheti, et al. *OpenVLA: An Open-Source Vision-Language-Action Model.* CoRL 2024. [arXiv:2406.09246](https://arxiv.org/abs/2406.09246)
12. <a id="ref-12"></a>R. Cadene, S. Alibert, F. Capuano, et al. *LeRobot: An Open-Source Library for End-to-End Robot Learning.* ICLR 2026. [arXiv:2602.22818](https://arxiv.org/abs/2602.22818)
13. <a id="ref-13"></a>M. Shukor, D. Aubakirova, F. Capuano, et al. *SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics.* arXiv preprint, 2025. [arXiv:2506.01844](https://arxiv.org/abs/2506.01844)

<details>
<summary>BibTeX for this page</summary>

```bibtex
@article{wilson1927probable,
  title         = {{Probable Inference, the Law of Succession, and Statistical Inference}},
  author        = {Edwin B. Wilson},
  journal       = {Journal of the American Statistical Association},
  volume        = {22},
  number        = {158},
  pages         = {209--212},
  year          = {1927},
  doi           = {10.1080/01621459.1927.10502953}
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

@article{gulcehre2023rest,
  title         = {{Reinforced Self-Training (ReST) for Language Modeling}},
  author        = {Caglar Gulcehre and Le Paine, Tom and Srivatsan Srinivasan and Ksenia Konyushkova and Lotte Weerts and Abhishek Sharma and Aditya Siddhant and Alex Ahern and Miaosen Wang and Chenjie Gu and Wolfgang Macherey and Arnaud Doucet and Orhan Firat and Nando de Freitas},
  journal       = {arXiv preprint arXiv:2308.08998},
  year          = {2023},
  eprint        = {2308.08998},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2308.08998}
}

@inproceedings{peters2007rwr,
  title         = {{Reinforcement Learning by Reward-Weighted Regression for Operational Space Control}},
  author        = {Jan Peters and Stefan Schaal},
  booktitle     = {International Conference on Machine Learning (ICML)},
  pages         = {745--750},
  year          = {2007},
  doi           = {10.1145/1273496.1273590}
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

@inproceedings{ho2020ddpm,
  title         = {{Denoising Diffusion Probabilistic Models}},
  author        = {Jonathan Ho and Ajay Jain and Pieter Abbeel},
  booktitle     = {Advances in Neural Information Processing Systems (NeurIPS)},
  year          = {2020},
  eprint        = {2006.11239},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2006.11239}
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

@inproceedings{zhao2023act,
  title         = {{Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware}},
  author        = {Tony Z. Zhao and Vikash Kumar and Sergey Levine and Chelsea Finn},
  booktitle     = {Robotics: Science and Systems (RSS)},
  year          = {2023},
  eprint        = {2304.13705},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2304.13705}
}

@inproceedings{kim2024openvla,
  title         = {{OpenVLA: An Open-Source Vision-Language-Action Model}},
  author        = {Moo Jin Kim and Karl Pertsch and Siddharth Karamcheti and Ted Xiao and Ashwin Balakrishna and Suraj Nair and Rafael Rafailov and Ethan Foster and Grace Lam and Pannag Sanketi and Quan Vuong and Thomas Kollar and Benjamin Burchfiel and Russ Tedrake and Dorsa Sadigh and Sergey Levine and Percy Liang and Chelsea Finn},
  booktitle     = {Conference on Robot Learning (CoRL)},
  year          = {2024},
  eprint        = {2406.09246},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2406.09246}
}

@inproceedings{cadene2026lerobot,
  title         = {{LeRobot: An Open-Source Library for End-to-End Robot Learning}},
  author        = {Remi Cadene and Simon Alibert and Francesco Capuano and Michel Aractingi and Adil Zouitine and Pepijn Kooijmans and Jade Choghari and Martino Russi and Caroline Pascal and Steven Palma and Mustafa Shukor and Jess Moss and Alexander Soare and Dana Aubakirova and Quentin Lhoest and Quentin Gallouédec and Thomas Wolf},
  booktitle     = {International Conference on Learning Representations (ICLR)},
  year          = {2026},
  eprint        = {2602.22818},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2602.22818}
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
```

</details>

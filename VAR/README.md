# Distilling Auto-regressive Models into Few Steps 1: Image Generation

**[[paper (arXiv)](https://arxiv.org/abs/2412.17153)]**
**[[code](https://github.com/imagination-research/distilled-decoding)]**
**[[website](https://imagination-research.github.io/distilled-decoding)]**

This is the example code for Distilled Decoding (DD) applied on [VAR](https://arxiv.org/abs/2404.02905) model. The code is heavily based on the [VAR codebase](https://github.com/FoundationVision/VAR).

--------------------

## Model zoo
We provide VAR-DD models for you to play with, which can be downloaded from the following links:

|   model    | reso. |     FID-1step      |     FID-2step      | #params | Link                     |
|:----------:|:-----:|:--------:|:-------:|:-------:|:------------------------------------------------------------------------------------|
|  VAR-DD-d16   |  256  |   9.94   |    7.82    |  327M   | [var-d16.pth](https://huggingface.co/microsoft/distilled_decoding/resolve/main/VAR-DD/var-d16.pth) |
|  VAR-DD-d20   |  256  |   9.55   |    7.33    |  635M   | [var-d20.pth](https://huggingface.co/microsoft/distilled_decoding/resolve/main/VAR-DD/var-d20.pth) |
|  VAR-DD-d24   |  256  |   8.92   |    6.95    |  1.09B  | [var-d24.pth](https://huggingface.co/microsoft/distilled_decoding/resolve/main/VAR-DD/var-d24.pth) |

## Commands

We provide commands to reproduce the FID results of few-step sampling and teacher-involved sampling in our paper. Before evaluating the performance, please download the VAE model from the original [VAR codebase](https://github.com/FoundationVision/VAR?tab=readme-ov-file#var-zoo). FID-statistic of ImageNet-256 can be downloaded at [this link](https://huggingface.co/microsoft/distilled_decoding/resolve/main/ImageNet-256_FID_Statistic/statistic.pth).

### Few-step sampling

#### d16 model
```
# 1step
python sample.py --ckpt_path /PATH/TO/VAR-D16/MODEL --vae_ckpt /PATH/TO/VAE --sample_dir outputs/fewstep_sample/d16/step1 --sample_num 50000 --t_list 0 --loss_split_t 5 --depth 16 --bs 64 --fid_statistic /PATH/TO/FID/STATISTIC/DIR

# 2step 
python sample.py --ckpt_path /PATH/TO/VAR-D16/MODEL --vae_ckpt /PATH/TO/VAE --sample_dir outputs/fewstep_sample/d16/step2 --sample_num 50000 --t_list 0,5 --loss_split_t 5 --depth 16 --bs 64 --fid_statistic /PATH/TO/FID/STATISTIC/DIR
```

#### d20 model
```
# 1step
python sample.py --ckpt_path /PATH/TO/VAR-D20/MODEL --vae_ckpt /PATH/TO/VAE --sample_dir outputs/fewstep_sample/d20/step1 --sample_num 50000 --t_list 0 --loss_split_t 4 --depth 20 --bs 64 --fid_statistic /PATH/TO/FID/STATISTIC/DIR

# 2step 
python sample.py --ckpt_path /PATH/TO/VAR-D20/MODEL --vae_ckpt /PATH/TO/VAE --sample_dir outputs/fewstep_sample/d20/step2 --sample_num 50000 --t_list 0,4 --loss_split_t 4 --depth 20 --bs 64 --fid_statistic /PATH/TO/FID/STATISTIC/DIR
```

#### d24 model
```
# 1step
python sample.py --ckpt_path /PATH/TO/VAR-D24/MODEL --vae_ckpt /PATH/TO/VAE --sample_dir outputs/fewstep_sample/d24/step1 --sample_num 50000 --t_list 0 --loss_split_t 4 --depth 24 --bs 64 --fid_statistic /PATH/TO/FID/STATISTIC/DIR


# 2step 
python sample.py --ckpt_path /PATH/TO/VAR-D24/MODEL --vae_ckpt /PATH/TO/VAE --sample_dir outputs/fewstep_sample/d24/step2 --sample_num 50000 --t_list 0,4 --loss_split_t 4 --depth 24 --bs 64 --fid_statistic /PATH/TO/FID/STATISTIC/DIR
```

### Teacher-involved sampling
Before teacher-involved, please download the VAR-d16 model from the original [VAR codebase](https://github.com/FoundationVision/VAR?tab=readme-ov-file#var-zoo). The expected results of teacher-involved sampling are listed below.

|   model    | reso. |     FID      |     step      | #params |
|:----------:|:-----:|:--------:|:-------:|:-------:|
|  VAR-DD-d16   |  256  |   6.54   |    3    |  327M   |
|  VAR-DD-d16   |  256  |   5.47   |    4    |  327M   |
|  VAR-DD-d16   |  256  |   5.03   |    6    |  327M   |

```
# 3step
python combine_sample.py --ckpt_path /PATH/TO/VAR-D16/MODEL --vae_ckpt /PATH/TO/VAE --sample_dir outputs/teacher_involved_sample/3step --sample_num 50000 --t_list 0,5 --loss_split_t 5 --depth 16 --bs 64 --teacher_path /PATH/TO/TEACHER/VAR-D16/MODEL --t_teacher_start 4 --fid_statistic /PATH/TO/FID/STATISTIC/DIR

# 4step
python combine_sample.py --ckpt_path /PATH/TO/VAR-D16/MODEL --vae_ckpt /PATH/TO/VAE --sample_dir outputs/teacher_involved_sample/4step --sample_num 50000 --t_list 0,5 --loss_split_t 8 --depth 16 --bs 64 --teacher_path /PATH/TO/TEACHER/VAR-D16/MODEL --t_teacher_start 3 --fid_statistic /PATH/TO/FID/STATISTIC/DIR

# 6step
python combine_sample.py --ckpt_path /PATH/TO/VAR-D16/MODEL --vae_ckpt /PATH/TO/VAE --sample_dir outputs/teacher_involved_sample/6step --sample_num 50000 --t_list 0,5 --loss_split_t 9 --depth 16 --bs 64 --teacher_path /PATH/TO/TEACHER/VAR-D16/MODEL --t_teacher_start 0 --fid_statistic /PATH/TO/FID/STATISTIC/DIR
```

## TODO

- [ ] Release the code of training data generation with flow-matching.
- [ ] Release the code of training.

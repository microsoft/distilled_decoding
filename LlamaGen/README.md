# Distilling Auto-regressive Models into Few Steps 1: Image Generation

**[[paper (arXiv)](https://arxiv.org/abs/2412.17153)]**
**[[code](https://github.com/imagination-research/distilled-decoding)]**
**[[website](https://imagination-research.github.io/distilled-decoding)]**

This is the example code Distilled Decoding (DD) applied on [LlamaGen](https://arxiv.org/abs/2406.06525) model. The code is heavily based on the [LlamaGen codebase](https://github.com/FoundationVision/LlamaGen).

--------------------

## Model zoo
We provide LamaGen-DD models for you to play with, which can be downloaded from the following links:

|   model    | reso. |     FID-1step      |     FID-2step      | #params | Link                     |
|:----------:|:-----:|:--------:|:-------:|:-------:|:------------------------------------------------------------------------------------|
|  LlamaGen-DD-B   |  256  |   15.50   |    11.17    |  98.3M   | [LlamaGen-B.pt](https://huggingface.co/microsoft/distilled_decoding/resolve/main/LlamaGen-DD/llamagen-B.pt) |
|  LlamaGen-DD-L   |  256  |   11.35   |    7.58    |  326M   | [LlamaGen-L.pt](https://huggingface.co/microsoft/distilled_decoding/resolve/main/LlamaGen-DD/llamagen-L.pt) |

## Commands

We provide commands to reproduce the FID results of few-step sampling and teacher-involved sampling in our paper. Before evaluating the performance, please download the VAE model from the original [LlamaGen codebase](https://github.com/FoundationVision/LlamaGen?tab=readme-ov-file#-class-conditional-image-generation-on-imagenet). FID-statistic of ImageNet-256 can be downloaded at [this link](https://huggingface.co/microsoft/distilled_decoding/resolve/main/ImageNet-256_FID_Statistic/statistic.pth).

### Few-step sampling

#### LlamaGen-B model
```
# 1step
python autoregressive/sample/eval_fewstep.py --vq-ckpt /PATH/TO/VAE --gpt-ckpt /PATH/TO/GPT-B/MODEL  --gpt-model GPT-B --image-size 256 --split_t 80 --bs 128 --save_path outputs/fewstep_sample/model_B/step1 --ema --t_list 0 --image_num 50000 --fid_statistic /PATH/TO/FID/STATISTIC/DIR

# 2step 
python autoregressive/sample/eval_fewstep.py --vq-ckpt /PATH/TO/VAE --gpt-ckpt /PATH/TO/GPT-B/MODEL  --gpt-model GPT-B --image-size 256 --split_t 80 --bs 128 --save_path outputs/fewstep_sample/model_B/step2 --ema --t_list 0,80 --image_num 50000 --fid_statistic /PATH/TO/FID/STATISTIC/DIR
```

#### LlamaGen-L model
```
# 1step
python autoregressive/sample/eval_fewstep.py --vq-ckpt /PATH/TO/VAE --gpt-ckpt /PATH/TO/GPT-L/MODEL  --gpt-model GPT-L --image-size 256 --split_t 130 --bs 256 --save_path outputs/fewstep_sample/model_L/step1 --ema --t_list 0 --image_num 50000 --fid_statistic /PATH/TO/FID/STATISTIC/DIR

# 2step 
python autoregressive/sample/eval_fewstep.py --vq-ckpt /PATH/TO/VAE --gpt-ckpt /PATH/TO/GPT-L/MODEL  --gpt-model GPT-L --image-size 256 --split_t 130 --bs 256 --save_path outputs/fewstep_sample/model_L/step2 --ema --t_list 0,80 --image_num 50000 --fid_statistic /PATH/TO/FID/STATISTIC/DIR
```

### Teacher-involved sampling
Before teacher-involved, please download the GPT-L model with sequence length 16x16 from the original [LlamaGen codebase](https://github.com/FoundationVision/LlamaGen?tab=readme-ov-file#-class-conditional-image-generation-on-imagenet). The expected results of teacher-involved sampling are listed below.

|   model    | reso. |     FID      |     step      | #params |
|:----------:|:-----:|:--------:|:-------:|:-------:|
|  LlamaGen-DD-L   |  256  |   6.76   |    22    |  326M   |
|  LlamaGen-DD-L   |  256  |   6.20   |    42    |  326M   |
|  LlamaGen-DD-L   |  256  |   5.71   |    81    |  326M   |

```
# 22step
python autoregressive/sample/eval_combine_sample.py --vq-ckpt /PATH/TO/VAE --gpt-ckpt /PATH/TO/GPT-L/MODEL  --gpt-model GPT-L --image-size 256 --split_t 180 --bs 128 --save_path outputs/teacher_involved_sample/22step --ema --gpu 1 --t_list 0,80 --image_num 50000 --teacher-ckpt /PATH/TO/TEACHER/GPT-L/MODEL --teacher_start_t 60 --fid_statistic /PATH/TO/FID/STATISTIC/DIR

# 42step
python autoregressive/sample/eval_combine_sample.py --vq-ckpt /PATH/TO/VAE --gpt-ckpt /PATH/TO/GPT-L/MODEL  --gpt-model GPT-L --image-size 256 --split_t 180 --bs 128 --save_path outputs/teacher_involved_sample/42step --ema --gpu 1 --t_list 0,80 --image_num 50000 --teacher-ckpt /PATH/TO/TEACHER/GPT-L/MODEL --teacher_start_t 40 --fid_statistic /PATH/TO/FID/STATISTIC/DIR

# 81step
python autoregressive/sample/eval_combine_sample.py --vq-ckpt /PATH/TO/VAE --gpt-ckpt /PATH/TO/GPT-L/MODEL  --gpt-model GPT-L --image-size 256 --split_t 220 --bs 128 --save_path outputs/teacher_involved_sample/81step --ema --gpu 1 --t_list 0,80 --image_num 50000 --teacher-ckpt /PATH/TO/TEACHER/GPT-L/MODEL --teacher_start_t 0 --fid_statistic /PATH/TO/FID/STATISTIC/DIR
```

## TODO

- [ ] Release the code of training data generation with flow-matching.
- [ ] Release the code of training.
- [ ] Release the code of text-to-image task.

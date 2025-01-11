# Distilled Decoding 1: One-step Sampling of Image Auto-regressive Models with Flow Matching

**[[paper (arXiv)](https://arxiv.org/abs/2412.17153)]**
**[[paper (ICLR25)](https://openreview.net/forum?id=zKlFXV87Pp&noteId=TXjgUIaXYP)]**
**[[code](https://github.com/imagination-research/distilled-decoding)]**
**[[website](https://imagination-research.github.io/distilled-decoding)]**


**Authors:** [Enshu Liu (Tsinghua)](https://scholar.google.com/citations?user=0LUhWzoAAAAJ&hl=en)\*, [Xuefei Ning (Tsinghua)](https://nics-effalg.com/ningxuefei/), [Yu Wang (Tsinghua)](https://scholar.google.com/citations?user=j8JGVvoAAAAJ&hl=en), [Zinan Lin (Microsoft Research)](https://zinanlin.me/)†

* \* Work mostly done during Enshu Liu's internship at Microsoft Research

* † Project advisor: Zinan Lin

This is the official repository of paper [Distilled Decoding 1: One-step Sampling of Image Auto-regressive Models with Flow Matching](https://arxiv.org/abs/2412.17153). We propose Distilled Decoding (DD) to distill a pre-trained image auto-regressive model to few steps for fast sampling.

We provide examples of applying DD to SOTA image AR models like [VAR](https://arxiv.org/abs/2404.02905) and [LlamaGen](https://arxiv.org/abs/2406.06525). Please refer to the contents in the directories [`./VAR`](VAR) and [`./LlamaGen`](LlamaGen) for more details. Currently, we only support sampling with our released DD models on ImageNet-256. We will release the training code and text-to-image models later.

## News
* `4/21/2025`: **The DD models and inference code for ImageNet-256 have been released!** The DD training code and text-to-image models are undergoing an additional Microsoft's internal review process and will be released at a later date.
  * Code: https://github.com/microsoft/distilled_decoding 
  * Models: https://huggingface.co/microsoft/distilled_decoding
* `12/24/2024`: The paper is released [here](https://arxiv.org/abs/2412.17153).
* `12/22/2024`: The project website is released [here](https://imagination-research.github.io/distilled-decoding).
* `12/22/2024`: The code and the pre-trained `DD` models are currently under Microsoft's internal review. We will release them here once the review is done. Please star/watch the repo to get the latest update.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

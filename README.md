# SuperGlobal

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2308.06954)



[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-features-are-all-you-need-for-image/image-retrieval-on-roxford-hard)](https://paperswithcode.com/sota/image-retrieval-on-roxford-hard?p=global-features-are-all-you-need-for-image)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/global-features-are-all-you-need-for-image/image-retrieval-on-rparis-hard)](https://paperswithcode.com/sota/image-retrieval-on-rparis-hard?p=global-features-are-all-you-need-for-image)



ICCV 2023 Paper *Global Features are All You Need for Image Retrieval and Reranking* Official Repository🚀🚀🚀


> Image retrieval systems conventionally use a two-stage paradigm, leveraging global features for initial retrieval and local features for
> reranking. However, the scalability of this method is often limited due to the significant storage and computation cost incurred by local
> feature matching in the reranking stage. In this paper, we present SuperGlobal, a novel approach that exclusively employs global features
> for both stages, improving efficiency without sacrificing accuracy. SuperGlobal introduces key enhancements to the retrieval system,
> specifically focusing on the global feature extraction and reranking processes. For extraction, we identify sub-optimal performance when the
> widely-used ArcFace loss and Generalized Mean (GeM) pooling methods are combined and propose several new modules to improve GeM pooling. In
> the reranking stage, we introduce a novel method to update the global features of the query and top-ranked images by only considering
> feature refinement with a small set of images, thus being very compute and memory efficient. Our experiments demonstrate substantial
> improvements compared to the state of the art in standard benchmarks. Notably, on the Revisited Oxford+1M Hard dataset, our single-stage
> results improve by 7.1%, while our two-stage gain reaches 3.7% with a strong 64,865x speedup. Our two-stage system surpasses the current
> single-stage state-of-the-art by 16.3%, offering a scalable, accurate alternative for high-performing image retrieval systems with minimal
> time overhead.


Leveraging global features only, our series of methods contribute to state-of-the-art performance in ROxford (+1M), RParis (+1M), and GLD test set with orders-of-magnitude speedup.

## Demo

![image](https://github.com/ShihaoShao-GH/SuperGlobal/blob/main/demo.gif)

## Results Reproduce

1) Download Revisited Oxford & Paris from https://github.com/filipradenovic/revisitop, and
save to path `./revisitop`.

2) Download CVNet pretrained weights from https://github.com/sungonce/CVNet, and save to path `./weights`.

3) Run 

`python test.py MODEL.DEPTH [50, 101] TEST.WEIGHTS ./revisitop TEST.DATA_DIR ./weights
SupG.gemp SupG.rgem SupG.sgem SupG.relup SupG.rerank`

And you will get the exact reported results in `log.txt`.

## Application

If you would like to try out our methods on other benchmarks or tasks, 
I recommend to go over `./modules` in this repository, and plug in your desired 
modules. They are very easy to use, and can be directly attached to your trained model!

## Acknowledgement

Many thanks to [CVNet](https://github.com/sungonce/CVNet), [DELG-pytorch](https://github.com/feymanpriv/DELG),
where we found resources to build our repository 
and they inspired us to have this work published!

## Contact us

Feel free to reach out our co-corresponding authors at shaoshihao@pku.edu.cn, and bingyi@google.com.
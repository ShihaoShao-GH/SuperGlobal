# SuperGlobal
ICCV' 23 Paper *Global Features are All You Need for Image Retrieval and Reranking* Official Repository🚀🚀🚀

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

Feel free to reach out our co-corresponding authors at shaoshihao@pku.edu.cn, bingyi@google.com.

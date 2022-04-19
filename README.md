##
This is the source code of paper ["Contrastive Graph Convolutional Networks with Adaptive Augmentation for Text Classification"]
<div align="center">    
<img src="https://github.com/yangyt-2020/CGA2TC/blob/main/src/model.png?raw=true" width="1000px" height="300px" alt="model.png" align=center />
</div>

## Require

* Python 3.6
* PyTorch 1.0
* pyg

## Running training and evaluation

1. `cd ./preprocess`
2. Run `python remove_words.py <dataset>`
3. Run `python build_graph.py <dataset> --aug 1 --PMI_size 0.01 --TF_IDF_size 0.01 --sample_size 1`
4. `cd ..`
5. Run `python train_supervised.py <dataset>`
6. Replace `<dataset>` with `R8`, `R52`, `ohsumed` or `mr`


## Cite


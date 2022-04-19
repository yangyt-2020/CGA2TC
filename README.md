

## Require

* Python 3.6
* PyTorch 1.0

## Running training and evaluation

1. `cd ./preprocess`
2. Run `python remove_words.py <dataset>`
3. Run `python build_graph.py <dataset>`
4. `cd ..`
5. Run `python train.py <dataset>`
6. Replace `<dataset>` with `20ng`, `R8`, `R52`, `ohsumed` or `mr`


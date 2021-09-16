# Tokenizer
A tokenizer using camembert tokenizer dictionary

We create this tokenizer based on Camembert Tokenizer for inferencing. In fact, 
using transformer tokenizer require to install Transformer and Torch/Tensorflow.

This tokenizer is useful for traning and mostly for inference. For your rest API, no need to install tensorflow or pytorch, you have just to convert your model to onnx format.

app.py give an example of using


Steps for using with tokenizer : 

1 ## Get à vocabulary
Get a vocabulary of your tokenizer via transformers package for example.
This code below give an example to get a vocabulary.

```python
!pip install transformers
!mkdir my_vocab

from transformers import CamembertFastTokenizer
tokenizer.save_vocabulary("./my_vocab/")
```

When your dictionary has dumped, you can it into sentencepiece like this
```python
!pip install sentencepiece
s = spm.SentencePieceProcessor(model_file = './my_vocab/sentencepiece.bpe.model')
``` 
It can be used to train your model. An example of usage with onnx have given in app.py
    

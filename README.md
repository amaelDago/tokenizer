# Tokenizer
A tokenizer using camembert tokenizer dictionary

We create this tokenizer based on Camembert Tokenizer for inferencing. In fact, 
using transformer tokenizer require to install Transformer and Torch/Tensorflow.

This tokenizer is useful for traning and mostly for inference. For your rest API, no need to install tensorflow or pytorch, you have just to convert your model to onnx format.

app.py give an example of using


Steps for using with tokenizer : 

##  1 Get camaembert vocabulary
Get a vocabulary of your tokenizer via transformers package for example.
This code below give an example to get a vocabulary.

```python
!pip install transformers
!mkdir my_vocab

from transformers import CamembertFastTokenizer
tokenizer = CamembertFastTokenizer.from_pretrained("camembert-base")
tokenizer.save_vocabulary("./my_vocab/")
```

## 2 Create your tokenizer
When your dictionary has dumped, you can put it into sentencepiece instance like this
```python
!pip install sentencepiece
s = spm.SentencePieceProcessor(model_file = 'my-sentencepiece-folder/sentencepiece.bpe.model')
``` 
It can be used to train your model. An example of API REST usage with onnx and flask has given in app.py
    

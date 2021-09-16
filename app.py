from flask import Flask, json, jsonify, request
import numpy
from onnxruntime import ExecutionMode, InferenceSession, SessionOptions

# Add infernece parameters
options = SessionOptions()
options.inter_op_num_threads = 2
options.execution_mode = ExecutionMode.ORT_SEQUENTIAL


def toNumpy(myList):
    # Trasnform list to numpy aarray
    return numpy.array(myList + [0] * (100 - len(myList))).reshape(1,-1)


def predict_sentiment(review) : 
    assert isinstance(review, str), "Error ! Input must be string" 

    tokens = tokenizer.encode_plus(review)
    print(tokens)

    tokens['input_ids'] = toNumpy(tokens['input_ids'])
    tokens["attention_mask"] = toNumpy(tokens["attention_mask"])
    tokens = {name : numpy.atleast_2d(value) for name, value in tokens.items()}
    tokens['input_ids'] = tokens['input_ids'].reshape(1, -1)
    tokens['attention_mask'] = tokens['attention_mask'].reshape(1, -1)
    output = session.run(None, tokens)
    ind = numpy.argmax(output)
    return ind

app = Flask(__name__)

@app.route('/')
def hello() : 
    return jsonify("hello")


@app.route("/predict", methods = ['POST'])
def predict() : 
    if request.method == "POST" : 
        req = request.get_data().decode("utf-8").strip()
        output = predict_sentiment(req.strip())
        res = f"The category of your document are : {output}"
        return jsonify(res)

if __name__ == "__main__" : 
    from my_vocab.mytokenize import Tokenizer
    import sentencepiece as spm

    #!mkdir my_vocab
    #tokenizer.save_vocabulary("./my_vocab/")
    s = spm.SentencePieceProcessor(model_file = './my_vocab/sentencepiece.bpe.model')
    tokenizer = Tokenizer(s)

        # Model
    # Instanciate a session inference
    session = InferenceSession("./sentiment-analysis-model.onnx", options)

    app.run(debug=True, host="0.0.0.0")

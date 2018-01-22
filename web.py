import argparse
from flask import Flask, render_template, request

from language_model import LM
from prediction import Model


app = Flask(__name__)
model = None  # type: Model


@app.route('/')
def main():
    phrase = request.args.get('phrase')
    top = []
    if phrase:
        tokens = tokenize(phrase)
        top = model.predict_top(tokens)
    return render_template(
        'main.html',
        phrase=phrase,
        top=top,
        )


def tokenize(phrase):
    return phrase.encode('utf8')


def run():

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    hps = LM.get_default_hparams().parse('num_steps=20,num_shards=8,num_layers=2,emb_size=12,projected_size=12,state_size=80,num_sampled=0,batch_size=1,vocab_size=102')
    hps._set("num_gpus", 1)
    #arg('model')
    #arg('vocab')
    arg('--port', type=int, default=8000)
    arg('--host', default='localhost')
    arg('--debug', action='store_true')
    args = parser.parse_args()

    global model
    #model = Model(args.model, args.vocab, hps)
    model = Model('/Users/ruiyangwang/Desktop/f-lm/logs/test/train/model.ckpt-0','/Users/ruiyangwang/Desktop/examples/word_language_model/data/penn/vocabulary.txt', hps)
    app.run(port=args.port, host=args.host, debug=args.debug)


if __name__ == '__main__':
    run()

from sacred import Experiment

ex = Experiment()

@ex.config
def config():
    foo = 33
    bar = {
        'fizz' : None,
    }

@ex.named_config
def py_named_1():
    py = {'value' : 1}
    
@ex.named_config
def py_named_2():
    py = {'value' : 2}

@ex.automain
def main(foo, bar, py, json, yaml):
    ...

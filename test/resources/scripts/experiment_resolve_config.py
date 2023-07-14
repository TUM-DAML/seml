
from sacred import Experiment

ex = Experiment()

@ex.config
def config():
    foo = 33
    bar = {
        'fizz' : None,
    }

@ex.automain
def main(foo, bar, external1, external2):
    ...

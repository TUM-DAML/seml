from sacred import Experiment

ex = Experiment()


@ex.automain
def main(foo, bar, param1, interpolated_1, interpolated_2):
    ...

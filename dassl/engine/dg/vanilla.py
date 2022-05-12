

from dassl.engine import TRAINER_REGISTRY



@TRAINER_REGISTRY.register()
class Vanilla():
    """Vanilla baseline."""

    def forward_backward(self, batch):
        pass
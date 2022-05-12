import torch




from dassl.engine import TRAINER_REGISTRY



@TRAINER_REGISTRY.register()
class CrossGrad():
    """Cross-gradient training.

    https://arxiv.org/abs/1804.10745.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
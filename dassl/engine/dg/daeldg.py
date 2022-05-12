





from dassl.engine import TRAINER_REGISTRY





















@TRAINER_REGISTRY.register()
class DAELDG():
    """Domain Adaptive Ensemble Learning.

    DG version: only use labeled source data.

    https://arxiv.org/abs/2003.07325.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
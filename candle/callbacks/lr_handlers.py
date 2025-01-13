from os.path import exists

from candle.callbacks import Callback


class LRTracker(Callback):
    def __init__(self):
        super().__init__()

    def before_training_starts(self):
        self.tracker.add_variable("lr", exists_ok=True)

    @staticmethod
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def on_epoch_end(self):
        self.tracker['lr'].update(self.get_lr(self.trainer.optimizer))

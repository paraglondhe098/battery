import time

from candle.callbacks import Callback


class DuplicateCallbackError(Exception):
    def __init__(self):
        super().__init__("ConsoleLogger is a unique callback and can only be added once!")


class ConsoleLogger(Callback):
    def __init__(self,
                 display_time_elapsed: bool = True,
                 round_off_upto: int = 5,
                 report_in_one_line: bool = True,
                 progress_bar_positions=("training",)):
        super().__init__(priority=float('-inf'))
        self.epoch_headline = "EPOCH {}"
        self.separator = "  ||  " if report_in_one_line else "\n"
        self.display_time_elapsed = display_time_elapsed
        self.roff = round_off_upto
        self.start_time = time.time()
        self.progress_bar_positions = progress_bar_positions

        self.__dashes = "-" * 100

    def before_training_starts(self):
        self.logger.info("-" * 45 + "Progress" + "-" * 45)
        self.trainer.progress_bar.positions.extend(self.progress_bar_positions)
        time.sleep(1)
        self.start_time = time.time()
        if not self.is_unique():
            raise DuplicateCallbackError

    def on_train_begin(self):
        self.trainer.epoch_headline = self.epoch_headline.format(self.trainer.current_epoch)

    def on_epoch_end(self):
        self.logger.info(self.tracker.message("--> Metrics: "))
        if self.display_time_elapsed:
            self.logger.info(f"Time elapsed: {time.time() - self.start_time} s")
        self.logger.info(self.__dashes)

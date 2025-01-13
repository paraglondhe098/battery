import logging
from abc import abstractmethod
from candle.utils.module import Module
from candle.callbacks import Callback, CallbacksList, ConsoleLogger
# from candle.callbacks import CallbackLambda
import torch
from typing import Optional, List, Callable, Any
from tqdm import tqdm


class TrainerModule(Module):
    def __init__(self, name: str,
                 device: Optional[torch.device] = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(name, device, logger)

    @abstractmethod
    def fit(self, X, Y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class ProgressBar:
    def __init__(self, positions, **kwargs):
        self.positions = positions
        self.kwargs = kwargs

    def __call__(self, position, iterable, desc):
        self.kwargs['desc'] = desc
        self.kwargs['iterable'] = iterable
        if position in self.positions:
            return tqdm(**self.kwargs)
        else:
            return iterable


class TrainerBlueprint(TrainerModule):
    default_callbacks = [ConsoleLogger()]

    def __init__(self,
                 callbacks: Optional[List[Callback]] = None,
                 clear_cuda_cache: bool = False,
                 use_amp: bool = False,
                 name=None,
                 device=None,
                 logger=None):
        super().__init__(name=name, device=(device or torch.device('cpu')), logger=logger)

        self.model = None
        self.final_metrics = {}
        self.__current_epoch = 0
        self.epochs = None
        self.epoch_headline = ""
        self.STOPPER = False
        self.best_state_dict = None
        self.tracker = self.init_tracker()

        callback_types = {type(callback) for callback in callbacks}
        for cb in self.default_callbacks:
            if type(cb) not in callback_types:
                callbacks.append(cb)

        self.callbacks = CallbacksList(callbacks or [], trainer=self)
        self.clear_cuda_cache = clear_cuda_cache
        self.use_amp = use_amp
        self.external_events = set()
        self.std_pos = {'on_train_batch_begin', 'on_train_batch_end', 'on_epoch_begin', 'on_epoch_end',
                        'on_test_batch_begin', 'on_test_batch_end', 'on_predict_batch_begin', 'on_predict_batch_end',
                        'on_train_begin', 'on_train_end', 'on_test_begin', 'on_test_end', 'on_predict_begin',
                        'on_predict_end', 'before_training_starts', 'after_training_ends', 'before_backward_pass'}

    @abstractmethod
    def init_tracker(self):
        pass

    def get_best_state_dict(self):
        return self.best_state_dict or self.get_state_dict()

    @abstractmethod
    def get_state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

    @abstractmethod
    def train(self, T: Any):
        pass

    @abstractmethod
    def validate(self, V: Any):
        pass

    def _run_callbacks(self, pos: str) -> List[Optional[str]]:
        return self.callbacks.run_all(pos)

    @property
    def current_epoch(self):
        return self.__current_epoch

    @abstractmethod
    def reset(self):
        pass

    def pre_run(self):
        pass

    def post_run(self):
        pass

    def fit(self, T: Any, V: Any, epochs: int = 10, epoch_start: int = 0):
        self.reset()
        self.pre_run()
        self.epochs = epochs
        on_gpu = True if self.device.type == 'cuda' else False

        self._run_callbacks(pos="before_training_starts")
        for self.__current_epoch in range(epoch_start, epoch_start + epochs):
            self._run_callbacks(pos="on_epoch_begin")
            if on_gpu and self.clear_cuda_cache:
                torch.cuda.empty_cache()

            self.train(T)
            self.validate(V)

            self._run_callbacks(pos="on_epoch_end")
            self.tracker.snap_and_reset_all()
            if self.STOPPER:
                break

        self._run_callbacks(pos="after_training_ends")
        self.post_run()
        return self.tracker.get_history()

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def save_progress(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_progress(self, *args, **kwargs):
        pass

    class __CallbackTemplate(Callback):
        def __init__(self):
            super().__init__()

    def add_event(self, pos: str):
        """
        Write a custom callback event without explicitly creating a new callback class.
        """

        def decorator(event: Callable) -> Optional[Callable]:
            if event.__name__ in self.external_events:
                return None

            ct = self.__CallbackTemplate()

            if pos in self.std_pos:
                setattr(ct, pos, event)
            else:
                raise AttributeError(f"Invalid method '{pos}'. Must be one of {self.std_pos}.")

            self.external_events.add(event.__name__)
            self.callbacks.append(ct)

            return event

        return decorator


# def dynamic_event(pos: str, priority: float = float('inf')):
#     def decorator(event: Callable) -> Callable:
#         def wrapper(instance, *args, **kwargs):
#             cb = CallbackLambda(pos=pos, func=event, priority=priority)
#             instance.callbacks.append(cb)
#             return event(instance, *args, **kwargs)
#         return wrapper
#     return decorator


class AdversarialTrainer(TrainerModule):
    def __init__(self):
        super().__init__(name="Adversarial Trainer")

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass


class LLMTrainer(TrainerModule):
    def __init__(self):
        super().__init__(name="LLM Trainer")

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass


class SemiSupervisedTrainer(TrainerModule):
    def __init__(self):
        super().__init__(name="Semi Supervised Trainer")

    def fit(self, X, Y):
        pass

    def predict(self, X):
        pass

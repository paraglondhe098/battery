from candle.callbacks.base import Callback


class WeightInitializer(Callback):
    def __init__(self):
        super().__init__()


class Regularizer(Callback):
    def __init__(self):
        super().__init__()


class LayerFreezer(Callback):
    def __init__(self):
        super().__init__()


class ImageSaver(Callback):
    def __init__(self):
        super().__init__()


class NotebookLogger(Callback):
    def __init__(self):
        super().__init__()


class TensorBoardLogger(Callback):
    def __init__(self):
        super().__init__()


class CSVLogger(Callback):
    def __init__(self):
        super().__init__()


class LRScheduler(Callback):
    def __init__(self):
        super().__init__()


class StateManager(Callback):
    def __init__(self):
        super().__init__()


class EarlyStopping(Callback):
    def __init__(self):
        super().__init__()


class LRTracker(Callback):
    def __init__(self):
        super().__init__()


class GradientClipping(Callback):
    def __init__(self):
        super().__init__()


class IntraEpochReport(Callback):
    def __init__(self):
        super().__init__()


class MemoryUsageLogger(Callback):
    def __init__(self):
        super().__init__()


class WeightWatcher(Callback):
    def __init__(self):
        super().__init__()


class ReduceLROnPlateau(Callback):
    def __init__(self):
        super().__init__()


class FeatureMapVisualizer(Callback):
    def __init__(self):
        super().__init__()


class RemoteMonitor(Callback):
    def __init__(self):
        super().__init__()


class NoiseInjector(Callback):
    def __init__(self):
        super().__init__()

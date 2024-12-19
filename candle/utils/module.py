from abc import ABC
import torch
import logging
from typing import Union, Optional


class Module(ABC):
    """Base class for all components of the framework.

    This class provides common functionality like device management, logging,
    and utility methods that are used across different components.

    Args:
        name (str, optional): Name to identify the module instance.
            Defaults to class name if not provided.
        device (torch.device, optional): Device to use (CPU/GPU).
            If None, device attribute will not be created.
    """

    def __init__(self, name: Optional[str] = None, device: Optional[torch.device] = None):
        self.name = name or self.__class__.__name__
        if device is not None:
            self.device = device

        # Initialize logger for this module instance
        self.logger = self._init_logger()

    def _init_logger(self) -> logging.Logger:
        """Initializes and configures a logger for the module.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(f"{self.name}")

        # Only add handlers if none exist to prevent duplicate logging
        if not logger.hasHandlers():
            # Create console handler with formatting
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

            # Set default level to INFO
            logger.setLevel(logging.INFO)

        return logger

    def to_device(self, obj: Union[torch.nn.Module, torch.Tensor]) -> Union[torch.nn.Module, torch.Tensor]:
        """Moves a PyTorch object to the module's device.

        Args:
            obj: PyTorch module or tensor to move to device

        Returns:
            The object moved to the appropriate device

        Raises:
            TypeError: If obj is not a PyTorch module or tensor
            AttributeError: If module doesn't have a device attribute
        """
        if not hasattr(self, 'device'):
            raise AttributeError(
                "Module instance has no device attribute. "
                "Ensure device was provided in __init__"
            )

        if isinstance(obj, (torch.nn.Module, torch.Tensor)):
            return obj.to(self.device)
        else:
            raise TypeError(
                f"Object of type {type(obj)} is not a torch.nn.Module or torch.Tensor"
            )

    def get_device_info(self) -> str:
        """Returns information about the currently used device.

        Returns:
            str: Description of current device
        """
        if hasattr(self, 'device'):
            if self.device.type == 'cuda':
                return f"Running on GPU: {torch.cuda.get_device_name(self.device)}"
            return f"Running on CPU"
        return "No device specified"

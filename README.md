
![Alt text](style/board_logo_new.png)

[//]: # (Candle: PyTorch Training Framework 🕯️)

A versatile PyTorch training framework to simplify and enhance the model training process.
It includes a trainer class with efficient training methods, 
famous built in pre-trained architectures, metrics tracking, custom and built-in 
callbacks support, and much more!

______________________________________________________________________

<div align="center">

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pytorch-candle)](https://pypi.org/project/pytorch-candle/)
[![PyPI Status](https://badge.fury.io/py/pytorch-candle.svg)](https://badge.fury.io/py/pytorch-candle)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pytorch-candle)](https://pepy.tech/project/pytorch-candle)

[//]: # ([![Conda]&#40;https://img.shields.io/conda/v/conda-forge/pytorch_candle?label=conda&color=success&#41;]&#40;https://anaconda.org/conda-forge/candle&#41;)
[//]: # ([![codecov]&#40;https://codecov.io/gh/paraglondhe098/pytorch-candle/graph/badge.svg?token=SmzX8mnKlA&#41;]&#40;https://codecov.io/gh/paraglondhe098/pytorch-candle&#41;)

![GitHub commit activity](https://img.shields.io/github/commit-activity/w/paraglondhe098/pytorch-candle)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

<a target="_blank" href="https://www.youtube.com/@Neuralogic-j8i">
  <img src="style/get_started_nn.png" height="36px" alt="Get started"/>
</a>
</div>

______________________________________________________________________

## Installation

Using pip:

```bash
    pip install pytorch-candle
```

Using conda:

```bash
    conda install pytorch-candle
```

<details>
  <summary>Advanced install options</summary>
    <!-- following section will be skipped from PyPI description -->
</details>

## Usage

### Trainer
Use trainer class to train your model with ease. Use custom model or use built-in models available in `candle.models` module.

Example: Training a simple CNN model on MNIST dataset.
```python
# Step 0: Import necessary modules
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from candle import Trainer
from candle.metrics import Accuracy, Precision
from candle.models.vision import BasicCNNClassifier
from candle.callbacks import EarlyStopping, ConsoleLogger

# Step 1: Initiate dataset and dataloader (See PyTorch documentation for more details)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Step 2: Define a Model using torch.nn.Module or use built-in models
# (stay tuned for more models in future releases.)
model = BasicCNNClassifier(input_shape = (1,64,64), num_output_classes=10)

# Step 3: Define metrics, loss function, optimizer, callbacks, etc. and pass them to Trainer
# (See documentation for more details on each parameter.)
accuracy = Accuracy(binary_output=False)
precision = Precision(binary_output=False)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
es = EarlyStopping(basis="val_accuracy", metric_minimize=False, patience=10, threshold=0.85)

# ConsoleLogger is a callback to display training progress in console along with progress bars.
cl = ConsoleLogger(display_time_elapsed=True,
                   round_off_upto=2,
                   report_in_one_line=True,
                   progress_bar_positions=["training", "validation", "prediction"])

trainer = Trainer( model,
                 criterion=loss_fn,
                 optimizer=optimizer,
                 metrics=[accuracy, precision],
                 callbacks=[es, cl],
                 device=torch.device('cuda'),
                 use_amp=True # Mixed precision training
                   )

# Step 3: Start training
history = trainer.fit(train_loader,val_loader, epochs=10)

# Step 4: Plot results and save progress
# (Metrics over epochs)
trainer.tracker.plot('accuracy', 'val_accuracy')
trainer.tracker.plot('loss', 'val_loss')

trainer.save_progress(path="path/to/save", metric_name="val_accuracy")
# Saves model, optimizer, scheduler, tracker, and other necessary objects in a folder.
# save folder will include final val_accuracy value (rounded upto 2 digits) in name (defaults to val_loss)
```

### Metrics

candle includes various metrics like `Accuracy`, `Precision`, `Recall` and `R² Score`, which can be used to evaluate model performance.
Defining custom metrics is also simplified.
```python
from candle.metrics import R2Score, Metric
import torch

class CustomMetric(Metric):
    def __init__(self):
        super().__init__(name = "your_metric_name", 
                         # It's important to give a name to the metric to access it using that name.
                         # By default, it's the class name.
                         pre_transform= lambda x: (x[0], x[1].squeeze()) 
                         # pre transform labels or outputs.
                         )
    def calculate(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
        # Overwrite this function to define custom metric logic.
        # Return average value for a batch (scalar output).
        pass
    
# Initialize the metric
r2_score = R2Score()
custom_metric  = CustomMetric()

X_batch, y_true = next(iter(val_loader))
y_pred = model(X_batch)

# Compute score
accuracy_score = r2_score(y_true, y_pred)
custom_score = custom_metric(y_true, y_pred)
```

### Callbacks

Callbacks allow you to add custom functionality during training, such as early stopping or saving model checkpoints at different intervals.
Creating custom callbacks is also supported.
```python
from candle.callbacks import EarlyStopping, Callback

# Create custom callbacks
early_stopping = EarlyStopping(basis='val_loss', patience=3, metric_minimize=True)
class CustomCallback(Callback):
    def __init__(self):
        super().__init__()
        ...
    
    def on_epoch_end(self):
        # Around 20 positions are reserved for callbacks
        ...
        
# Add callback to the trainer
trainer = Trainer(..., callbacks=[early_stopping, CustomCallback()])
```
Callbacks have access to trainer, metrics tracker, stopping main training loop,
latest metric values, learning rate, etc. at around 20 pre-defined positions including
before_training_starts, after_training_ends, on_epoch_begin, on_train_batch_begin, etc. Just
overwrite these functions to access desired positions in training process.

### Tracker

Tracker manages the storing, averaging, and reporting the metrics including loss
over batches. This tracker object can be used by callback objects to gain acess to history
of metrics over epoch which is stored in a records list. Tracker object can also be used
to retrieve latest metric value at any point of time like during training, validation,
after training, etc. Tracker object also have a method to plot metric values over time.
```python
from candle import Trainer

trainer = Trainer(**kwargs)

history = trainer.fit(train_loader,val_loader, epochs=10)

trainer.tracker.plot('accuracy', 'val_accuracy')
trainer.tracker.plot('loss', 'val_loss')

print("latest accuracy value: " + trainer.tracker['accuracy'].latest)
print("accuracy value at 20th epoch was: " + trainer.tracker['accuracy'][20])

accuracy_link = trainer.tracker.create_link("accuracy", split = "child")
print("Latest accuracy (linked) value: ",accuracy_link.latest)
# Linking as child helps to aggregate metrics over different times (time less than an epoch) to track them
# at different periods (per batch or per k-batches).
# Other modes: "clone" and "self" (read docs).
```


## Contributing

We welcome contributions! To contribute, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Version

Current version: `0.0.16`

[//]: # (## Contact)

[//]: # ()
[//]: # (For any questions or inquiries, contact me at `paraglondhe123`.)

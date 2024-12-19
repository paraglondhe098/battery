Candle: PyTorch Training Framework 🔥

This repository provides a versatile PyTorch training framework to simplify and enhance the model training process. It includes a trainer class with efficient training methods, famous built in pre-trained architectures, metrics tracking, custom and built-in callbacks support, and much more!

## Installation

Using pip:

```bash
    pip install pytorch-candle
```

Using conda:

```bash
    conda install pytorch-candle
```

## Usage

### Trainer


```python
from candle import Trainer
import torch
from candle.metrics import Accuracy
from candle.models.vision import BsicCNN
from candle.callbacks import EarlyStopping, IntraEpochReport

model = BsicCNN(input = (1,28,28), output = (10,))
accuracy = Accuracy(binary_output=False)
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
es = EarlyStopping(**es_kwargs)
ier = IntraEpochReport(**ier_kwargs)

trainer = Trainer( model,
                 criterion=loss_fn,
                 input_shape=(1,28,28),
                 optimizer=optimizer,
                 display_time_elapsed=True,
                 metrics=[accuracy],
                 callbacks=[es, ier],
                 device=torch.device('cuda'),
                 use_amp=True)

train_loader = ...
val_loader = ...

# Start training
history = trainer.fit(train_loader,val_loader, epochs=10)

trainer.tracker.plot('accuracy', 'val_accuracy')
```

### Metrics

candle includes various metrics like `Accuracy` and `R² Score`, which can be used to evaluate model performance.

```python
from candle.metrics import Accuracy, R2Score

# Initialize the Accuracy metric
accuracy = Accuracy()

# Compute accuracy
accuracy_score = accuracy(y_true, y_pred)
```

### Callbacks

Callbacks allow you to add custom functionality during training, such as early stopping or model checkpoints.

```python
from candle.callbacks import StateManager, Callback

# Create custom callbacks
early_stopping = StateManager(monitor='val_loss', patience=3)
class CustomCallback(Callback):
    def __init__(self):
        super().__init__()
        ...
    
    def on_epoch_end(self):
        ...
        
# Add callback to the trainer
trainer.add_callback(CustomCallback())
```


## Contributing

We welcome contributions! To contribute, please fork the repository, make your changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Version

Current version: `1.0.0`

[//]: # (## Contact)

[//]: # ()
[//]: # (For any questions or inquiries, contact me at `paraglondhe123`.)

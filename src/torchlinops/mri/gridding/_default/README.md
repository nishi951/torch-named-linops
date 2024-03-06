# Default implicit grog backend

## Layout
- `engine`
  - Training and evaluation code (generic)
  - Variants are nested inside at levels appropriate to their level of complexity
- `app`
  - For fully-featured plug-and-play "apps" for performing igrog
  - `basic`
    - implicit grog based on k-space calibration data
  - `image_based`
    - implicit grog based on image-domain data
  - `fieldcorr`
    - field corrected grog
- `config`
  - Standard configurations for variants inside `app`.
    - Mostly used to help scripts with configuring igrog
- `arch`
  - Neural network components

## Backend Organization (engine)
trainer: Training loop (generic/abstract/interface) - Apply data to improve the model
- Configurable factors include:
  - Type of optimizer/scheduler
    - Parameters of those optimizers and schedulers
- Responsibility:
  - Training the model with data

inferencer: Inference module (abstract/interface) - Use the model to get whatever you were wanting to get
- Configurable factors include:
  - Custom test-time behavior for the model
  - Partial validation
- Responsibility:
  - Providing an interface to run the model on data without training

datamodule: Data management (abstract/interface) - Manage the data and 
- Configurable factors include:
  - Which datasets to load
  - Which preprocessing to apply
  - Which postprocessing to apply
- Responsibility : 
  - Managing data formats, saving and loading

### To implement a new type of implicit GROG
Use the trainer and inferencer, combined with
- A new datamodule:
  - `preprocess`: Ingest the data and do any preprocessing required
    - Returns self
  - `postprocess`: Take in the output of applying the trained model to the test_dataloader and use it to update/grid the data.
    - Returns self
  - `train_dataloader()` : Create the implicit grog training dataset and dataloader 
    - Returns a pytorch DataLoader
  - `test_dataloader()` : Create the dataset on which the implicit GROG model will be applied
    - Returns a pytorch DataLoader
  - Outputs/modified stuff goes in the `.data` attribute
- A Loss Function (nn.Module)
  - Jointly coordinate with the datamodule on API/other behavior
- A Model (nn.Module)
  - Jointly coordinate with the loss function and datamodule on input/output behavior.

### Training Callbacks (hooks)
Philosophically, callbacks exist to allow external functions to inspect, update, or alter the training state
without manually making those functions dependencies of the trainer itself.
Examples of callbacks include:
- Updating the global step and the epoch
- Logging the loss, or other data every iteration
- Performing validation runs
- Clipping gradients

Note that this implementation of "callback" could includes essentially every
function that is performed in the usual course of a training iteration (e.g.
model forward pass, backward pass, loss, stepping the optimizer, etc.).
Basically, anything currently performed as a training step could also be
performed in a callback. This means things are fairly redundant right now but
perhaps this is preferable than making it so that there's only one way of doing
things in the future...

Callbacks include:
- `train_started`, `train_ended`
- `epoch_started`, `epoch_ended`
- `train_step_started`, `train_step_ended`

Callbacks must implement the `AbstractTrainingCallback` interface. They
themselves are stateless with regard to the training procedure, but may edit the
training state (this is to avoid random stateful behavior invisible to the
training loop). However, they may have additional attributes (such as loggers)
that take in data from the current training state and emit it in some form. The
training state itself may change and evolve, or be overridden to add additional
useful data that should be maintained over the course of the training process.

Finally, callbacks may have dependencies (specified by adding the names of the
dependent callback classes in the `dependencies` field). A callback will only be
called after all of it's dependencies have been called. 



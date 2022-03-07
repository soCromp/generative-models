# generative-models
Adapted from https://github.com/AntixK/PyTorch-VAE

``python run.py -m conf/model/vae.yaml -d conf/data/celeba.yaml``

**Requirements**
- Python 3.8.5
- PyTorch 1.10
- TorchVision 0.11
- PyTorch Lightning 1.5
- Torch-fidelity 0.3

**Files**
- *run.py*: Specify the model and hyperparameters, it creates, trains and tests the model.
- *experiment.py*: General wrapper through which other code interacts with specific kinds of models
- *vae.py*: Specification of the vanilla VAE model (later there will be one file per model, eg gan.py, diffusion_model.py)
- *dataloader.py*: Handles data preparations

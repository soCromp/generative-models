# generative-models
Adapted from https://github.com/AntixK/PyTorch-VAE

``conda env create -f conda_reqs.yml`` <br/>
``python run.py -m conf/model/vae.yaml -d conf/data/celeba.yaml``

**Requirements**
- Conda 4.12
- Python 3.8.5

**Files**
- *run.py*: Specify the model and hyperparameters, it creates, trains and tests the model.
- *model.py*: General wrapper through which other code interacts with specific kinds of models
- *vae.py*: Specification of the vanilla VAE model (later there will be one file per model, eg gan.py, diffusion_model.py)
- *dataloader.py*: Handles data preparations

Note: to update requirements

``conda env export > conda_reqs.yml``

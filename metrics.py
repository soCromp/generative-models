def inception_score():
    raise NotImplementedError


def frechet_score():
    raise NotImplementedError

"""
Need to modify these, which always expect 3-channel images unlike MNIST
https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/image/inception.py#L28-L171
https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/image/fid.py#L127-L276
"""

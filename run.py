import argparse
import vae

parser = argparse.ArgumentParser(description='Generic experiment driver')
parser.add_argument('--model',  '-m',
                    help =  'which model to use. Options: vae',
                    default='vae')
args = parser.parse_args()

if args.model == 'vae':
    print('will run vanilla VAE experiment')
    model = vae()
else:
    raise NotImplementedError

# interpret given hyperparameters

# set up model

# set up datasets

# do training

# do testing

import os

sizes = [10**i for i in range(1,5)] + [0] #0 corresponds to full dataset of 70k images
print(sizes)
with open('conf/data/mnist.yaml', 'r') as f:
    text = f.readlines()

for i in range(2): #trials
    for s in sizes:
        text[5] = f'num_samples: {s}\n'
        print(text)
        with open(f'conf/data/mnist{s}.yaml', 'w') as f:
            f.writelines(text)
        os.system(f"python run.py -m conf/model/vae.yaml -d conf/data/mnist{s}.yaml")

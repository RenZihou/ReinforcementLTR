# -*- encoding: utf-8 -*-
# @Author: RenZihou

epochs = 100
batches = 10

from tqdm import trange, tqdm
from time import sleep

pbar = tqdm(range(100))
for char in pbar:
    if char % 10:
        pbar.set_postfix({'char': char ** 2})
    sleep(0.03)

if __name__ == '__main__':
    pass

# -*- encoding: utf-8 -*-
# @Author: RenZihou

import numpy as np
import pandas as pd
from tqdm import tqdm


class MSLRDataset(object):
    """MSLR Dataset"""

    def __init__(self, path, features=136):
        # self.data = pd.DataFrame(columns=['qid', 'features', 'label'])
        self.data = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f'Loading {path}'):
                label, qid, *feats = line.strip().split()
                self.data.append({
                    'qid': np.int32(qid[4:]),
                    'feats': np.pad(
                        np.array([np.float32(f.split(':')[-1]) for f in feats[:features]]),
                        (0, max(0, features - len(feats))),
                    ),
                    'label': np.int32(label)})
        self.data = pd.DataFrame(self.data).reset_index(drop=True)

    def generate_group_per_query(self):
        """generate batch per query"""
        qids = self.data['qid'].unique()
        np.random.shuffle(qids)
        for qid in qids:
            group = self.data[self.data['qid'] == qid]
            yield np.stack(group['feats'].values), np.stack(group['label'].values)


if __name__ == '__main__':
    dataset = MSLRDataset('data/MSLR-WEB10K/test.txt')
    for x, y in dataset.generate_group_per_query():
        print(x, y)
    pass

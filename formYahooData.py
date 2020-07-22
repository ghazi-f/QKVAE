import pandas as pd
import numpy as np
from collections import defaultdict

# df = pd.read_csv('.data\yahoo\ydata-ynacc-v1_0_unlabeled_conversations.tsv', sep='\t', encoding='UTF-8',)
# print(df.head())
with open('.data\yahoo\ydata-ynacc-v1_0_unlabeled_conversations.tsv', encoding='UTF-8') as f:
    header = f.readline().split('\t')
    lines = []
    for line in f:
        if len(line.split('\t')) == len(header):
            lines.append(line.split('\t'))

print(header)
print(np.unique([len(l) for l in lines]))

comment_dict = defaultdict(list)
for line in lines:
    comment_dict[line[0]].append(line[9])
print([(k, v) for k, v in list(comment_dict.items())[:3]])

for split in ['train', 'dev', 'test']:
    with open('.data\yahoo\ydata-ynacc-v1_0_{}-ids.txt'.format(split)) as sf:
        with open('.data\yahoo\{}.txt'.format(split), 'w', encoding='UTF-8') as tf:
            not_found, found = 0, 0
            for line in sf:
                for comment in comment_dict[line.strip()]:
                    tf.write(comment+'\n')
                if len(comment_dict[line.strip()]) == 0:
                    not_found+=1
                else:
                    found+=1

            print('{} not found: '.format(split), not_found)
            print('{} found: '.format(split), found)


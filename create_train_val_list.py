import glob
import os

BASE_DIR = './data/isbi2015raw/'

def printlen(x):
    print(len(x))

def extract_casename(x):
    return os.path.split(x)[1].split('.')[0]


all_slices = sorted(glob.glob(BASE_DIR+'data/*.h5'))
print(len(all_slices))
# pprint(all_slices)
test_case = '03_05'

test_slices = [extract_casename(x) for x in all_slices if f'patient{test_case}' in x]
train_slices = [extract_casename(x) for x in all_slices if f'patient{test_case}' not in x and 'patient' in x]
printlen(test_slices)
printlen(train_slices)

# unlabeled_slices = sorted(glob.glob(BASE_DIR+'test/*.h5'))
# unlabeled_slices = map(extract_casename, unlabeled_slices)

# with open('./data/isbi2015raw/unlabeled_slices.list', 'w') as f:
#     for l in unlabeled_slices:
#         f.write(l)
#         f.write('\n')

with open(f'./data/isbi2015raw/train_slices_{test_case}.list', 'w') as f:
    for l in train_slices:
        f.write(l)
        f.write('\n')

with open(f'./data/isbi2015raw/test_{test_case}.list', 'w') as f:
    for l in test_slices:
        f.write(l)
        f.write('\n')

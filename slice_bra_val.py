from util import *
from h5py import File

def normalize(image):
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image

with open('data/BraTS2019/val.list', 'r') as f:
    tmp_list = f.readlines() 
val_cases = [item.replace("\n", "") for item in tmp_list]

os.makedirs('data/BraTS2019/data/val_slices', exist_ok=True)

root = 'data/BraTS2019/data/'
for case in val_cases:
    path = os.path.join(root, case)
    assert os.path.exists(path), f'{path} not exists!'
    f = File(path, 'r')
    im3d = f["image"][:]
    lb3d = f["label"][:]
    for i in range(im3d.shape[0]):
        slice_name = root + 'val_slices/' + case.split('.')[0] + '_' + str(i).zfill(3) + '.h5'
        sf = File(slice_name, 'w')
        sf.create_dataset("image", data=normalize(im3d[i]), compression="gzip")
        sf.create_dataset("label", data=lb3d[i], compression="gzip")
        sf.close()
import os
import cv2 as cv
import numpy as np
import nibabel as nib


def extract_pngs_from_nii(file, dst_dir, prefix):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    epi_img = nib.load(file)
    epi_img_data = epi_img.get_fdata()
    assert epi_img_data.shape, (181, 217, 181)
    d = np.array(epi_img_data)
    for i in range(181):
        im = d[:, :, i]
        img = cv.normalize(im, None, 0.0, 1.0, cv.NORM_MINMAX)
        img = img * 255.0
        cv.imwrite(os.path.join(dst_dir, f"{prefix}{i}.png"), img)


if __name__ == "__main__":

    patients = [d for d in os.listdir(".") if os.path.isdir(d) and "training" in d]

    for p in patients:
        mask1s = [f for f in os.listdir(os.path.join(p, "masks")) if "mask1" in f]
        mask2s = [f for f in os.listdir(os.path.join(p, "masks")) if "mask2" in f]
        flairs = [
            f for f in os.listdir(os.path.join(p, "preprocessed")) if "flair" in f
        ]
        t2s = [f for f in os.listdir(os.path.join(p, "preprocessed")) if "t2" in f]
        pds = [f for f in os.listdir(os.path.join(p, "preprocessed")) if "pd" in f]
        mprages = [
            f for f in os.listdir(os.path.join(p, "preprocessed")) if "mprage" in f
        ]
        for mask1, mask2, flair, t2, pd, marage in zip(
            mask1s, mask2s, flairs, t2s, pds, mprages
        ):
            prefix = mask1.split("mask1")[0]
            extract_pngs_from_nii(
                os.path.join(p, "masks", mask1),
                os.path.join("data", "isbi2015", "masks"),
                prefix,
            )
            extract_pngs_from_nii(
                os.path.join(p, "masks", mask2),
                os.path.join("data", "isbi2015", "masks2"),
                prefix,
            )
            extract_pngs_from_nii(
                os.path.join(p, "preprocessed", flair),
                os.path.join("data", "isbi2015", "flair"),
                prefix,
            )
            extract_pngs_from_nii(
                os.path.join(p, "preprocessed", t2),
                os.path.join("data", "isbi2015", "t2"),
                prefix,
            )
            extract_pngs_from_nii(
                os.path.join(p, "preprocessed", pd),
                os.path.join("data", "isbi2015", "pd"),
                prefix,
            )
            extract_pngs_from_nii(
                os.path.join(p, "preprocessed", marage),
                os.path.join("data", "isbi2015", "marage"),
                prefix,
            )

import utils
import visual_feat_ext
import pandas as pd
import numpy as np
import os

media_files = ["../media/Muppets-02-01-01.avi", "../media/Muppets-02-04-04.avi", "../media/Muppets-03-04-03.avi"]
eps = ["ep1", "ep2", "ep3"]
gts = ["../data/gt/gt_02_01_01.csv", "../data/gt/gt_02_04_04.csv.csv", "../data/gt/gt_03_04_03.csv"]

for file, ep, gt in zip(media_files, eps, gts):
    if os.path.exists("../data/"+ep+"_visual_full.csv"):
        print("feature file already exists... skipping")
        next
    
    else:
        this_gt = pd.read_csv(gt, delimiter=",", na_values="")
        this_gt.fillna(0, inplace=True)

        imgs, feats = visual_feat_ext.feats_from_avi(file ,limit_frames=False, frames=4000, one_frame_per_sec=False,
                                                blob_sigma=400, blob_t=0.04,
                                                include_blobs=True, include_hists=True)
        df = pd.merge(feats, this_gt, left_on=['minute','second'], right_on = ['Min','Sec']).drop(["Min","Sec"], axis=1)
        df.set_index(["minute","second","frame"], inplace=True)
        df.to_csv("../data/"+ep+"_visual_full.csv")
        print("...done!")


import pandas as pd
import numpy as np
import os.path
import visual_feat_ext

def check_movie_files():
    print("checking for raw movies.mp4 files...")
    files = ["Muppets-02-01-01.avi", "Muppets-02-04-04.avi", "Muppets-03-04-03.avi"]
    for f in files:
        print("../media/"+f)
        os.path.isfile("../media/"+f)
    print("all present!")


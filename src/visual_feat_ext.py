import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from skimage.feature import blob_doh
from tqdm import tqdm

def feats_from_avi(fpath="../media/Muppets-02-04-04.avi",
                   limit_frames=True, frames=100, one_frame_per_sec=False,
                  blob_sigma=400, blob_t=0.04,
                  include_blobs=True, include_hists=True):
    """
        Return features from movie file. Features are Blobs and/or color histograms.
        Blob detection can be specified by sigma and threshold parameter t. TODO: add green masks as param
        Frames can be limited to achieve quicker results while debugging.
        Function returns a pandas dataframe with features as columns.
    """
    print("processing file:", fpath, "...")
    vcap = cv2.VideoCapture(fpath)
    success,image = vcap.read()
    nr_frames = int(vcap.get(cv2.CAP_PROP_FRAME_COUNT))
    if limit_frames:
        frms = frames
    elif not limit_frames:
        frms = nr_frames
    #count = 0 
    mins = -1
    minflip = False
    seconds = -1
    imgs = []
    hists = []
    blobs = pd.DataFrame()
    feats = pd.DataFrame(columns=["minute", "second", "frame",
                                 # "blobs",
                                 # "hists"
                                 ])

    for count in tqdm(range(0, frms)):

        # handle frame to time stuff
        if (count%25)==0:
            seconds +=1
        if (seconds%60)==0:
            if not minflip:
                minflip=True
                mins+=1
        elif (seconds%60)!=0 & minflip:
            minflip=False
        
        # to make this whole thing quicker, add ability to only get 1 frame per second...
        if one_frame_per_sec==False:
            # get blobs -- this is multi array with (x,y,radius), potentially many blobs, so safe in dataframe 
            if include_blobs:
                thisblob = pd.DataFrame(return_green_blobs(image, sigma=blob_sigma, t=blob_t))
                blobs = pd.concat((blobs, thisblob), axis=1, ignore_index=True)
            if include_hists:
                hists.append( return_hists(image) )
            imgs.append(image) # BGR
            # append feature blobs to dataframe for current minute, second, frame
            feats = feats.append({"minute":mins, "second": seconds%60, "frame": count%25+1,
                                 # "blobs": blobs,
                                 # "hists": hists, 
                                 }, ignore_index=True)
            
        elif one_frame_per_sec==True and (count%25)==0:
            if include_blobs:
                thisblob = pd.DataFrame(return_green_blobs(image, sigma=blob_sigma, t=blob_t))
                blobs = pd.concat((blobs, thisblob), axis=1, ignore_index=True)
            if include_hists:
                hists.append( return_hists(image) )
            imgs.append(image) # BGR
            # append feature blobs to dataframe for current minute, second, frame
            #print(mins, seconds, count%25+1)
            feats = feats.append({"minute":mins, "second": seconds%60, "frame": count%25+1,
                                 # "blobs": blobs,
                                 # "hists": hists, 
                                 }, ignore_index=True)
            
        success,image = vcap.read()
        count += 1
    
    histDF = pd.DataFrame(hists)
    blobDF = blobs.transpose()
    
    resDF = pd.concat((feats, histDF.add_suffix("_hist")), axis=1)
    resDF = pd.concat((resDF, blobDF.add_suffix("_blob")), axis=1)
    
    return(imgs, resDF)

def return_green_blobs(image, plot=False, sigma=400, t=0.04):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(image_hsv)
    #plt.imshow(image[:,:,::-1])

    # define range of green color in HSV for kermit
    lower_green = np.array([30,50,50])
    upper_green = np.array([65,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(image_hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

    #print(res)
    #plt.imshow(res[:,:,1], cmap="gray")
    
    hue, sat, val = cv2.split(res)
    retval, thresholded = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #plt.imshow(thresholded, cmap="gray")

    medianFiltered = cv2.medianBlur(thresholded,27)
    #plt.imshow(medianFiltered, cmap="gray")

    blobs_doh = blob_doh(medianFiltered, max_sigma=sigma, threshold=t)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(8, 7), sharex=True, sharey=True)
        ax = axes.ravel()
        ax[0].imshow(image[:,:,::-1])

        ax[1].imshow(medianFiltered, cmap="gray")
        for blob in blobs_doh:
            y, x, r = blob
            c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
            ax[1].add_patch(c)
    
    return(blobs_doh.flatten())


def return_hists(image, plot=False):
    img = image[:,:,::-1].copy() # BGR to RGB
    color = ('r','g','b')
    hists = []
    for i in range(0,3):
        hists.append(cv2.calcHist([img], [i], None, [256], [0,256]).flatten()) 
    #hists = cv2.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256]).flatten()
    hists = np.array(hists)
    if plot:
        plt.figure(figsize=(10,5))
        plt.subplot(1,2,1)
        plt.imshow(img)
        for i, col in enumerate(color):
            plt.subplot(1,2,2)
            plt.plot(hists[i],color = col)
            plt.xlim([0,256])
        plt.show()
    hists = np.stack(hists).flatten()
    return(hists)

# Intermediate Hand-In: Finding Kermit with Audio & Video Features
The intermediate Hand in consists only of the kermit task for now...

## Source code
```src/__debug_visual_kermit.ipynb``` ... detailed information on how visual features were created 
```src/audio_kermit_tpot_optimization.ipynb``` ... finding optimal audio classifier via AutoML (TPOT API) 
```src/audio_kermit.ipynb``` ... main pipeline for audio features (mfcc) 
```src/audio_waldorf_statler.ipynb``` ... train a TPOT on MFCC data for w+s with CV 
```src/create_visual_feature_csv.py``` ... creates visual feature csv files 
```src/utils.py``` ... the famous utils.py file every project needs
```src/visual_feat_ext.py``` ... helper functions for visual feature extraction (blob detection & color histogramm) 
```src/visual_kermit.ipynb``` ... kermit visual pipeline, feature creation and prediction on train/test split // CV
```src/visual_waldorf_statler.ipynb``` ... train a TPOT classifier based on (already created) histogram data for w+s detection
```src/roc_figures/``` ... contains roc_figures for classifiers
```src/tpot_exports/``` ... contains exported tpot pipelines with best classifiers

## Complete student data
please find our complete student data attached in ```student_data.txt```
## Entry point of the code (e.g. main Matlab file)
We recommend you to download our repository from [our github repository](https://github.com/oStritze/similarity-modeling), since we could not provide our feature extraction csv files due to file size.  You could also extract the features by executing ```src/audio_kermit.ipynb``` and ```src/create_visual_features_csv.py```.

After extracting the features / downloading them, you are able to follow the implementation in the respective notebooks for Kermit / WS Audio and Video pipelines. A debugging overview of how the video features were created can be found in ```__debug_visual_kermit.ipynb``` and ```audio_kermit.ipynb```. 

## Performance indicators (e.g. Recall, Precision, etc.)
We used ROC-AUC plots and SKlearns Precision, Recall, Accuracy and F1. As a optimization metric for TPOT AutoML we used ROC-AUC. Please find the ROC-AUC plots in ```src/roc_figures/```.

## Timesheets
### Gabriel
| Date | Time | Description |
--- | --- | ---
| 09 11 20 | 1200-1300 | Watch and annotate first episode |
| 13 11 20 | 1100-1600 | Create First Prototype |
| 20 11 20 | 1500-1800 | Working on Feature Extraction |
| 30 11 20 | 1000-1400 | Model Optimization |
| 02 12 20 | 1300-1400 | Watch and annotate third episode |
| 02 12 20 | 1400-1800 | Create evaluation pipeline |
| 05 12 20 | 2000-2100 | Finalizing intermediate hand-in |
| 11 01 21 | 1000-1800 | Kermit Audio Experimentation |
| 12 01 21 | 1100-1200 | Waldorf + Statler Video Implementation |

### Oliver
| Date | Time | Description |
--- | --- | ---
| 04 11 20 | 1000-1200 | Prepare Project and Git structure |
| 04 11 20 | 1300-1500 | getting in touch with openCV |
| 09 11 20 | 0800-0900 | Watch and annotate first episode |
| 14 11 20 | 1400-1800 | first tries with blob and histogram detection |
| 02 12 20 | 0800-1500 | extract visual features |
| 02 12 20 | 1600-1900 | train first classifers on visual features |
| 03 12 20 | 0800-1400 | extract visual features, train tpots |
| 04 12 20 | 0800-1500 | train pipelines, evaluate classifiers  |
| 06 12 20 | 1000-1200 | move results to evaluation pipeline, merge results from audio and video |
| 06 12 20 | 1200-1400 | finalizing intermediate hand-in |
| 31 12 20 | 0930-1330 | creating missing w+s ground truth |
| 31 12 20 | 1400-1500 | try kermit_video on cv rather than train-test-val |
| 31 12 20 | 1500-1700 | dig into existing audio features, trying first tpot models |
| 02 01 21 | 1400-1700 | trying different tpot approaches that fit into laptop memory |
| 03 01 21 | 1100-1600 | finalizing fitting into laptop memory |
| 14 01 21 | 1630-1800 | prettify existing folder structures |
| 14 01 21 | 1800-1900 | finalize kermit_video cv approach |
| 15 01 21 | 0900-1100 | finalize ws_audio approach, prettify |


## Infos on architecture, features, classifier, etc. - whatever you consider important/helpful
We decided to go for a supervised approach, so we had to watch and annotate the episodes ourselves. 
Since checking every single frame would have been too tedious, we opted to annotate seconds only. (annotations can be found in ```data/gt/```).  
### Audio
We opted for classical feature engineering and extracted mel-frequency-cepstral-coefficents and also delta- and delta-delta-mfccs.  
We flattened all of the matrices to gain a 2640-dimensional vector per second.  

### Video
Again, opting for classical feature engineering. First we had a look at green blobs in the picture, which were exported per frame. Additionally, we exported color histogram values. The exportation took a while per video file, and the paramters and code can be found in ```src/__debug_visual_feat_ext.py```. We tried classifiers for histograms, blobs and combining histogram and blobs, with last performing best. Details can be found in ```src/visual_kermit.ipynb```. 

## Test data (no videos, but images/audio with ground truth) & Weka Experimenter log file for classifier comparison (if applicable)
We split our data in 3 sets (train / test/ validation), where training happened on episode 02-01-01, testing on 02-04-04 and validation on 03-04-03.
Our manually annotated data can be found at ```data/gt/```.

__Final Hand in__: We tried a combined Cross validation approach and it performed reasonably better in terms of scoring metrics which we implemented for the final hand in for all classifiers. 

## Classifier performance (auc)
More Metrics can be found in the corresponding notebooks.

### Kermit
| Feature | Train | Validation |
--- | --- | ---
| MFCC | TODO | .79 |
| Blob & Histogram | 1.0 | 1.0 |

### Waldorf + Statler (auc)
| Feature | Train | Validation |
--- | --- | ---
| MFCC | 0.79 | .63 |
| Histogram | TODO | 1.0 |

## Interpetation of Results
From our results we conclude that the visual features are more promising in terms of predicting the target characters. Blob-detection for visual features had a positive impact, needing much more computational effort though when creating the features. Even for detecting Waldorf and Statler, the visual features (Color Histograms solely) appeared more promising than the audio feature extaction methods used (MFCC). 
Also, combining the episodes features and labels and creating a randomized train-test split over all episodes turned out as the better approach. This is problably due to the last episode being an outlier (Robin-Hood-Themed) with many occurences of green shaped and colored frames and overlapping audio/voice characteristics. 

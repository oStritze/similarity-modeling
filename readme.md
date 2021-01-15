# Intermediate Hand-In: Finding Kermit with Audio & Video Features
The intermediate Hand in consists only of the kermit task for now...

## Source code
```src/audio.ipynb``` ... main pipeline for audio features (mfcc)
```src/audio_tpot.py``` ... optimal audio classifier as found by TPOT  
```src/visual_feat_ext.py``` ... helper functions for visual feature extraction (blob detection & color histogramm)  
```src/create_visual_feature_csv.py``` ... creates visual feature csv files  
```src/evaluation.ipynb``` ... combines and evaluates audio and video features  
## Complete student data
please find our complete student data attached in ```student_data.txt```
## Entry point of the code (e.g. main Matlab file)
We recommend you to download our repository from [our github repository](https://github.com/oStritze/similarity-modeling), since we could not provide our feature extraction csv files due to file size. In this case you could use ```src/evaluation.ipynb``` as entry point.  
You could also extract the features by executing ```src/audio.ipynb``` and ```src/create_visual_features_csv.py``` and then look at ```src/evaluation.ipynb```.  
## Performance indicators (e.g. Recall, Precision, etc.)
For this intermediate exercise we only used precision as performance indicators.  
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



## Infos on architecture, features, classifier, etc. - whatever you consider important/helpful
We decided to go for a supervised approach, so we had to watch and annotate the episodes ourselves. 
Since checking every single frame would have been too tedious, we opted to annotate seconds only. (annotations can be found in ```data/gt/```).  
### Audio
We opted for classical feature engineering and extracted mel-frequency-cepstral-coefficents and also delta- and delta-delta-mfccs.  
We flattened all of the matrices to gain a 2640-dimensional vector per second.  
Then we used TPOT to find the optimal model, achieving a 92.3 accuracy score with an ensemble of Naive Bayes and Random Forest. The exported model can be found in ```src/audio_tpot.py```.  
### Video
Again, opting for classical feature engineering. First we had a look at green blobs in the picture, which were exported per frame. Additionally, we exported color histogram values. The exportation took a while per video file, and the paramters and code can be found in ```src/visual_feat_ext.py```. We tried classifiers for histograms, blobs and combining histogram and blobs, with last performing best. Details can be found in ```src/visual.ipynb```. 
### Ensemble (work in progress)
For the combined workflow, see ```src/evaluation.ipynb```.
We will build an ensemble combining both audio and video features after they separately annotate the data using their respective extracted features.  
For the intermediate hand-in we will use both predictions (audio & video) connected by logical and & logical or and use these as final predictions to compare against the val set.  
## Test data (no videos, but images/audio with ground truth) & Weka Experimenter log file for classifier comparison (if applicable)
We split our data in 3 sets (train / test/ validation), where training happened on episode 02-01-01, testing on 02-04-04 and validation on 03-04-03.
Our manually annotated data can be found at ```data/gt/```.
## Classifier performance (precision)

| Feature | Train | Test | Validation |
--- | --- | --- | ---
| MFCC | .9385 | .9231 | .6666 |
| Blob | .8677 | .8417 | .7210 |
| Histogram | - | .9117  | .3441 |
| Blob & Histogram | .9080 | .9209 | 0.8145 |
| All Combined (AND) | - | - | 1. (very bad recall though) |
| All Combined (OR) | - | - | 0.8193 |

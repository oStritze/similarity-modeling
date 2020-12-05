# Intermediate Hand-In: Finding Kermit with Audio & Video Features
...
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
### Oliver
| Date | Time | Description |
--- | --- | ---
| 01 01 20 | 1500-1700 | Klowasser trinken |
## Infos on architecture, features, classifier, etc. - whatever you consider important/helpful
...
## Test data (no videos, but images/audio with ground truth) & Weka Experimenter log file for classifier comparison (if applicable)
...
## ROC figures of classifier performance
...
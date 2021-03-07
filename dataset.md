## Data Set Information

* Audio track (encoded as mp3) of each of the 106,574 tracks. It is on average 10 millions samples per track.
* Nine audio features (consisting of 518 attributes) for each of the 106,574 tracks.
* Given the metadata, multiple problems can be explored: recommendation, genre recognition, artist identification, year prediction, music annotation, unsupervized categorization.
* Please see the paper and the GitHub repository for more information ([Web Link])


## Attribute Information

Nine audio features computed across time and summarized with seven statistics (mean, standard deviation, skew, kurtosis, median, minimum, maximum): 
1. Chroma, 84 attributes 
2. Tonnetz, 42 attributes 
3. Mel Frequency Cepstral Coefficient (MFCC), 140 attributes 
4. Spectral centroid, 7 attributes 
5. Spectral bandwidth, 7 attributes 
6. Spectral contrast, 49 attributes 
7. Spectral rolloff, 7 attributes 
8. Root Mean Square energy, 7 attributes 
9. Zero-crossing rate, 7 attributes


* features.csv : can be used in classification (use a subset)
* echonest.csv : temporal features (?)

The useful .ipynb according to Salvatore are:

* usage
* analysis
* baselines (starting classification models)

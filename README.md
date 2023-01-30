# Voice Processing for binary classification

Audio from participants in the study was analyzed for binary classification.

### Processing audio files

Audio files were processed using `audio_processing.ipynb`. Using transcripts form interviews, fragments were participants speak are selected and original audio files are sliced to contain only those fragments. The new audio files are saved as .wav files
Alternatively, those new audio files are further processed to be sliced into 10 seconds fragments

### Predicting labels using COVAREP feature data

COVAREP (https://covarep.github.io/covarep/) feature extraction was provided in the dataset for each participant in 10 millisecond fragments. 

The data from all participants was averaged and joined together into a new dataframe and stored into a new .csv file using `dataprocessing_COVAREP.ipynb`

#### COVAREP data visualization and model selection

The file `data_visualization_COVAREP.ipynb` contains visualization for COVAREP features data and model selection. 

Features from COVAREP were selected by observing correlation with label data, keeping highly correlated features.

Dimensionality reduction was performed using Principal Component Analysis (PCA)

Transformed data was used to evaluate model for classification: Logistic Regression, Random Forest and Gradient Boosting

### Convolutional Neural Network for binary classification of spectrograms

`nn_model.ipynb` shows training of a convolutional neural network for binary classification of spectrograms of the whole audio recording of every participant. 

`nn_model_10s.ipynb` shows training of a convolutional neural network for binary classification of spectrograms of the 10 seconds fragments of audio voice recordings from each participant. This model achieved the best performance. 

`nn_model_VGG19.ipynb` was an attempt at transfer learning from pre-trained VGG19. Better computing power is needed to further evaluate this approach.

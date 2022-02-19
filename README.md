# Synthetic-ATR-Dataset_Eval:
Evaluate encoder-decoder network using ATR dataset

# Task:
1.) Classify ground truth images (Print out classiification)
2.) Create synthetic images using different orientation (Print out image), then classify synthetic image (Print out classification)
3.) Find exact ground truth of different orientation, then classifiy ground truth of different orientation (Print out classifcation)
4.) Evaluate:
    - How well the synthetic images are compared to the ground truth
    - Accuracy of classifcations

# Requirements:
- 9 Classifications:
    - PICKUP (#1)
    - SUV    (#2)
    - BTR70  (#5)
    - BRDM2  (#6)
    - BMP2   (#9)
    - T72    (#11)
    - ZSU23  (#12)
    - 2S3    (#13)
    - D20    (#14)
    NOTE: T62 (#10) or human class will not be use.

- 2 Sensor:
    - Visible (ilco)
    - MWIR    (cegr)

- 5 degree offset

- 3 Ranges
    - 1000m (Day: 2003/Night: 1923)
    - 1500m (Day: 2005/Night: 1925)
    - 2000m (Day: 2007/Night: 1927)

# Image Comparison Method:
- Manifold Representation (t-SNE)

# Folder Structure
- Functions
    - process_classifier_data.py
        - Description: Helper function to process Train_Classifer_Dataset
    - process_encoder_decoder_data.py
        - Description: Helper function to process Train_Encoder_Decoder_Dataset
    - test_classifier.py
        - Description: Function to evaluate classifer performance
    - test_encoder_decoder.py
        - Description: Function to evaluate encoder decoder performance
    - train_classifier.py
        - Description: Function to train classifier
    - train_enoder_decoder.py
        - Description: Function to train encoder decoder
- INPUT
    - Train_Classifier_Dataset
        - Description: Dataset used to train the classifer.
    - Train_Encoder_Decoder_Dataset
        - Description: Dataset used to train encoder decoder
- OUTPUT
    - Description: Contains output of each run
- main.py
    - Description: Main function that controls training or evaluation
- README.md
    - Description: This file.
- .gitignore
    - Description: Ignore specific files

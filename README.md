# Segmentation of Pneumothorax on Chest CTs  Using Deep Learning Based on Unet-Resnet-50 Convolutional Neural Network Structure


## Code Description
    To develop a deep learning CNN model we create a structured python code. 
    This code includes 4 part because of limitation on RAM, GPU and others resources.
    Part 1 and 4 run local python environment. 
    Part 2 and 3 is time and resources consuming codes so we will run them in Google Colabratory 
environment with TPU backend. 
    
## Code Structure:
    ### Part 1/4: [PreprocessFiles] 
        Preprocess dicom, nifti and csv file
        Outputs: data.csv, dicomStore.npy and maskStore.npy
    ### Part 2/4: [PreprocessData] 
        Preprosess train_test split (Google Compute Engine)
        Outputs: x_train, y_train, x_test, y_test
    ### Part 3/4: [Model] 
        Build, train and test model. (Google Compute Emgine)
        Outputs: Model metrics and logs
    ### Part 4/4: [EDA] 
        Exploratory data analysis on data and model
        Outputs: Statistical table and graphs

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 09:26:32 2023

Segmentation of Pneumothorax on Chest CTs 
Using Deep Learning Based on Unet-Resnet-50 Convolutional Neural Network Structure

@author: drademgencer


Code Description
    To develop a deep learning CNN model we create a structured python code. 
    This code includes 4 part because of limitation on RAM, GPU and others resources.
    Part 1 and 4 run local python environment. 
    Part 2 and 3 is time and resources consuming codes so we will run them in Google Colabratory 
environment with TPU backend. 
    
Code Structure:
    Part 1/4: [PreprocessFiles] 
        Preprocess dicom, nifti and csv file
        Outputs: data.csv, dicomStore.npy and maskStore.npy
    Part 2/4: [PreprocessData] 
        Preprosess train_test split (Google Compute Engine)
        Outputs: x_train, y_train, x_test, y_test
    Part 3/4: [Model] 
        Build, train and test model. (Google Compute Emgine)
        Outputs: Model metrics and logs
    Part 4/4: [EDA] 
        Exploratory data analysis on data and model
        Outputs: Statistical table and graphs
        
"""

# PART 1 : PreprocessFiles


# Import required libraries
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import deep learning libraries
from sklearn import preprocessing as preprocess

# Import libraries for reading dicom files
import pydicom
import nibabel as nib


# Import os libraries
from os import listdir

# LOG
print("LOG: Libraries imported")


"""
    NEPTUNE
    initialization not required for this part
"""


"""
    Directory and File settings
"""

# Define path for files
# !!! Sensitive Data !!! please update these variables according to your needs.
baseDicomDir = '../Dataset/Dicom'               
saveDicomPath = 'outputs_local/dicomStore.npy'
baseMaskDir = '../Dataset/Mask'
saveMaskPath = 'outputs_local/maskStore.npy'
dfPath = '../Dataset/labels.csv'
saveDfPath = 'outputs_local/dataFrame.csv'


# Read from csv file
df = pd.read_csv(dfPath, sep=";")

# Modify df
df['Id'] = df['Id'].apply(lambda x: str(x).zfill(10))     # Repair ID's
df["dicom_dir"] =  baseDicomDir + "/" + df.Id + "/"
df["mask_dir"] = baseMaskDir + "/" + df.Id + "/"
df["volume_size"] = df.dicom_dir.apply(lambda x: len(listdir(x)))


# LOG
print("Directories initialized")


"""
    Scan file function
"""

# Scan files
def scan_files(pId):
    dfPatient = df.loc[df['Id'] == pId]
    dicomDirectory = dfPatient.dicom_dir.values[0]
    maskDirectory = dfPatient.mask_dir.values[0]
    
    # Mask data
    img = nib.load(maskDirectory + "/Segmentation.nii")
    masks = np.array(img.get_fdata())
    
    # Dicom data
    dFiles = [pydicom.dcmread(dicomDirectory + "/" + fname) for fname in tqdm(listdir(dicomDirectory))]
    dFiles.sort(key = lambda x: float(x.InstanceNumber))
    dFiles = np.array(dFiles)
    
    # Add Dicom data to dataframe base on first file
    df.loc[df['Id'] == pId, "rows"] = dFiles[0].Rows
    df.loc[df['Id'] == pId, "columns"] = dFiles[0].Columns
    df.loc[df['Id'] == pId, "area"] = dFiles[0].Rows * dFiles[0].Columns
    df.loc[df['Id'] == pId, "pixelspacing_r"] = dFiles[0].PixelSpacing[0]
    df.loc[df['Id'] == pId, "pixelspacing_c"] = dFiles[0].PixelSpacing[1]
    df.loc[df['Id'] == pId, "pixelspacing_area"] = dFiles[0].PixelSpacing[0] * dFiles[0].PixelSpacing[1]
    df.loc[df['Id'] == pId, "slice_thickness"] = dFiles[0].SliceThickness
    
    # HU Values
    dicoms = np.empty([512, 512,0])
    print("\nwaiting for hu calculation (#" + pId + ")...")
    for dFile in dFiles:
        # Get Pixel values
        pix = np.array(dFile.pixel_array)
        # Get hu values
        huv = pix * dFile.RescaleSlope + dFile.RescaleIntercept
        huv = np.expand_dims(huv, (-1)) 
        
        # Apply windows
        
        
        # Append hu values
        dicoms = np.append(dicoms, huv, axis=-1)
        
        
    # Normalization
    print("Normalizing data")
    dicoms = preprocess.MinMaxScaler().fit_transform(dicoms.reshape(-1,1)).reshape(dicoms.shape)
        
    # LOG
    print("Patient files #{} scanned".format(pId))
    print("Shape of mask: {}".format(masks.shape))
    print("Shape of dicoms: {}".format(dicoms.shape))
    print("next-->\n")

    return masks, dicoms

# LOG
print("Scan function initialized")


"""
    Searching Files
"""

# LOG
print(" ")
print("\n\n------------------------")
print("-    Start searching   -")
print("------------------------\n\n")

# Create store
maskStore   = np.empty([512, 512, 0])
dicomStore   = np.empty([512, 512, 0])


# Loop for patients
pIds = df.loc[:, 'Id'].values
for pId in tqdm(pIds):

    # Search files
    masksPt, dicomsPt = scan_files(pId)
      
    maskStore = np.append(maskStore, masksPt, axis=-1)
    dicomStore = np.append(dicomStore, dicomsPt, axis=-1)

# Repare Image Orientation
dicomStore = np.rot90(dicomStore, axes=(1,0))
dicomStore = np.fliplr(dicomStore)

# LOG
print("\n\n------------------------")
print("-    End of search     -")
print("------------------------\n\n")

# Check shapes
print("Shape of mask store {}".format(maskStore.shape))
print("Shape of dicom store {}".format(dicomStore.shape))


"""
    Save Files for Further Use
"""

np.save(saveDicomPath, dicomStore)
np.save(saveMaskPath, maskStore)
df.to_csv(saveDfPath, index=False)


# LOG
print("Files saved")





# LOG
print("---END OF CODE---")

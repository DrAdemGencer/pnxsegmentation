# PART 2 : PreprocessData


# Import required libraries
import numpy as np

# Import deep learning libraries
from sklearn.model_selection import train_test_split

# LOG
print("Libraries imported")


"""
    NEPTUNE
    initialization not required for this part
"""

"""
    Directory and File settings
"""

# Define path for files
# !!! Sensitive Data !!! please update these variables according to your needs.

datasetPath = '/content/drive/MyDrive/Colab Notebooks/Pnx Deep Learning/Dataset/'
dicom_data_path = datasetPath + 'dicomStore.npy'      
mask_data_path = datasetPath + 'maskStore.npy'
save_x_train_path = datasetPath + 'xTrain.npy'
save_y_train_path = datasetPath + 'yTrain.npy'
save_x_val_path = datasetPath + 'xVal.npy'
save_y_val_path = datasetPath + 'yVal.npy'
save_x_test_path = datasetPath + 'xTest.npy'
save_y_test_path = datasetPath + 'yTest.npy'

"""
    Split Train & Validation & Test data
"""

# Get data
xData = np.load(dicom_data_path)
yData = np.load(mask_data_path)

# Prepeare data for split
print("Transpozing data")
xData = xData.T
yData = yData.T

# Reshape
print("Reshaping data")
xData = np.expand_dims(xData, (-1)) 
yData = np.expand_dims(yData, (-1)) 

# Check shapes
print("\n\nShape before split")
print("Shape of xData {}".format(xData.shape))
print("Shape of yData {}".format(yData.shape))

# Train test split after shuffle (64%-16%-20%)
print("Apply train test split")
X_train, x_test, Y_train, y_test = train_test_split(xData, yData, test_size=0.2, shuffle=True, random_state=43)
x_train, x_val, y_train, y_val = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True, random_state=43)

# Save files for further use
np.save(save_x_train_path, x_train)
np.save(save_y_train_path, y_train)
np.save(save_x_val_path, x_val)
np.save(save_y_val_path, y_val)
np.save(save_x_test_path, x_test)
np.save(save_y_test_path, y_test)
print("Train & Val & Test Files saved")

# Check shapes
print("\n\nShape after split")
print("Shape of x_train {}".format(x_train.shape))
print("Shape of y_train {}".format(y_train.shape))
print("Shape of x_val {}".format(x_val.shape))
print("Shape of y_val {}".format(y_val.shape))
print("Shape of x_test {}".format(x_test.shape))
print("Shape of y_test {}".format(y_test.shape))

# LOG
print("---END OF CODE---")

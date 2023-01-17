# PART 3 : Model


# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle

# Import deep learning libraries
import tensorflow as tf
import segmentation_models as sm

# Import Neptune
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

# LOG
print("Libraries imported")


"""
    NEPTUNE INITIALIZATION
"""
# Run code for neptune
run = neptune.init_run(
    project="dr.ademgencer/unet-segmentation",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjZjAxYzMzZS1mMTVlLTQwZGQtOGMxZC05NDUwODk2ZDdlN2UifQ==",
)  # your credentials

neptune_cbk = NeptuneCallback(run=run)

print("Neptune initialized")


"""
    Directory and File settings
"""

# Data paths
# !!! Sensitive Data !!! please update these variables according to your needs.

datasetPath = '/content/drive/MyDrive/Colab Notebooks/Pnx Deep Learning/Dataset/'
x_train_path = datasetPath + 'xTrain.npy'
y_train_path = datasetPath + 'yTrain.npy'
x_val_path = datasetPath + 'xVal.npy'
y_val_path = datasetPath + 'yVal.npy'
x_test_path = datasetPath + 'xTest.npy'
y_test_path = datasetPath + 'yTest.npy'
predictions_path = datasetPath + 'predicted.npy'

# Model paths
# !!! Sensitive Data !!! please update these variables according to your needs.
modelPath = '/content/drive/MyDrive/Colab Notebooks/Pnx Deep Learning/Model/'
checkpoint_path = modelPath + "checkpoints/"
save_path = modelPath + "packages/"
graph_path = modelPath + "graphs/"
log_path = modelPath + "logs/"

# LOG
print("Directories initialized")

# Get data
x_train = np.load(x_train_path)
y_train = np.load(y_train_path)
x_val = np.load(x_val_path)
y_val = np.load(y_val_path)
x_test = np.load(x_test_path)
y_test = np.load(y_test_path)

# LOG
print("Data loaded")

"""
    Model construction
"""

# Model Backbone
model_backbone = 'resnet50'

# Model Preprocessing
preprocess_input = sm.get_preprocessing(model_backbone)

x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# Optimizer
model_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# loss function
model_loss = sm.losses.bce_jaccard_loss

# Metrics
model_metrics = [
    tf.keras.metrics.AUC(),
    tf.keras.metrics.BinaryAccuracy(),
    tf.keras.metrics.BinaryIoU(),
    tf.keras.metrics.MeanIoU(num_classes=2),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.FalseNegatives(),
    sm.metrics.FScore(), # Dice Coeff
    sm.metrics.IOUScore() # Jackard Index
]

# Callbacks
model_callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=8, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1),
    tf.keras.callbacks.TensorBoard(log_dir=log_path),
    neptune_cbk # NEPTUNE
]


EPOCHS = 1000
BATCH_SIZE = 32


# Define model
model = sm.Unet(model_backbone, encoder_weights=None, input_shape=(None, None, x_train.shape[-1]))

# Compile model
model.compile(
    optimizer = model_optimizer,
    loss = model_loss,
    metrics = model_metrics,
)

# Print summary
# print(model.summary())

# Plot model
tf.keras.utils.plot_model(model, to_file=(graph_path +'model_architecture.jpg'), show_shapes=True)

# Save model
timestr = time.strftime("%Y%m%d-%H%M%S")
model.save(save_path + 'pretrain-' + timestr + '.h5')

# LOG
print("Model constructed and saved")


"""
    Perform Deep Learning Model
    !!! This action takes too much time !!!
"""

# Fit model
history = model.fit(
    x = x_train,
    y = y_train,
    batch_size = BATCH_SIZE,
    epochs=EPOCHS,
    validation_data = (x_val, y_val),
    callbacks = model_callbacks,
)


# Save model
timestr = time.strftime("%Y%m%d-%H%M%S")
model.save(save_path + '/posttrain-' + timestr + '.h5', overwrite=True, include_optimizer=True)

# Save history
with open(log_path + 'trainHistoryDict' + timestr, 'wb') as file_pi:
    pickle.dump(history.history, file_pi)
print("History saved")

# LOG
print("Model trained!")
    
# Stop Neptune
run.stop()
print("Neptune stopped")

# Evaluate Model

# Get model /content/drive/MyDrive/Colab Notebooks/Pnx Deep Learning/Model/packages/posttrain-20230105-105634.h5
model = tf.keras.models.load_model(save_path + 'posttrain-20230105-105634.h5', compile=False)

# Compile model
model.compile(
    optimizer = model_optimizer,
    loss = model_loss,
    metrics = model_metrics,
)

# Evaluate
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)

print("Metric Results:", results)

print("Evaluation completed!")

# Predict model
predictions = model.predict(x=x_test, verbose=1)

# Save predictions
np.save(predictions_path, predictions)

# LOG
print("Prediction completed!")

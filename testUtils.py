import numpy as np
import pandas as pd
import time

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Conv2DTranspose, Concatenate, concatenate, AveragePooling2D, UpSampling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.image import extract_patches


# Custom callback for testing the model each 10 epochs
class TestCallback(keras.callbacks.Callback):
    """
    Callback for testing the model each test_freq epochs during training.
    Given a test_generator, the minimum and maximum values of the data,
    the function test_model is called to evaluate the model.
    The results are saved in the test_filepath file.
    """
    def __init__(self, test_generator, data_min, data_max, test_filepath, test_freq=10, test_steps=100):
        self.test_generator = test_generator
        self.test_filepath = test_filepath
        self.test_freq = test_freq
        self.data_min = data_min
        self.data_max = data_max
        self.test_steps = test_steps


    def on_epoch_end(self, epoch, logs=None):
        # epoch starts from 0 so we add 1
        if (epoch + 1) % self.test_freq == 0:
            test_model(self.model, self.test_generator,
                        self.data_min, self.data_max, 
                        self.test_filepath, self.test_steps)

class TestCallbackGaussian(keras.callbacks.Callback):
    """
    Callback for testing the model each test_freq epochs during training that uses the gaussian normalization.
    """
    def __init__(self, test_generator, data_mean, data_std, test_filepath, test_freq=10, test_steps=100):
        self.test_generator = test_generator
        self.test_filepath = test_filepath
        self.test_freq = test_freq
        self.data_mean = data_mean
        self.data_std = data_std
        self.test_steps = test_steps


    def on_epoch_end(self, epoch, logs=None):
        # epoch starts from 0 so we add 1
        if (epoch + 1) % self.test_freq == 0:
            test_model_gaussian(self.model, self.test_generator,
                                self.data_mean, self.data_std, 
                                self.test_filepath, self.test_steps)
            

class TestCallbackDincaeGaussian(keras.callbacks.Callback):
    """
    Callback for testing the model each test_freq epochs during training that uses the gaussian normalization and dincae generator.
    """
    def __init__(self, test_generator, data_mean, data_std, test_filepath, len_dataset, batch_size = 32, test_freq=10 ):
        self.test_generator = test_generator
        self.test_filepath = test_filepath
        self.test_freq = test_freq
        self.data_mean = data_mean
        self.data_std = data_std
        self.len_dataset = len_dataset
        self.batch_size = batch_size


    def on_epoch_end(self, epoch, logs=None):
        # epoch starts from 0 so we add 1
        if (epoch + 1) % self.test_freq == 0:
            test_dincae_gaussian(self.model, self.test_generator,
                                self.data_mean, self.data_std, 
                                self.test_filepath, self.len_dataset, self.batch_size)
        

# Custom reduce lr on plateau callback
# Basically the same but with a different wait parameter
# and more logs about the wait and the best value parameters
class customReduceLROnPlateau(tf.keras.callbacks.ReduceLROnPlateau):
    def __init__(self, old_wait=None,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        if old_wait is None:
            self.old_wait = self.wait
        else:
            self.old_wait = min(old_wait + 2, self.wait)


    def on_train_begin(self, logs=None):
        self.wait = self.old_wait


    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        # print(f'Epoch {epoch + 1}: current {self.monitor} = {current}, best = {self.best}, wait = {self.wait}')

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                if old_lr > self.min_lr:
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        print(f'\nEpoch {epoch + 1}: ReduceLROnPlateau reducing learning rate to {new_lr}.')
                    self.wait = 0


    def on_train_end(self, logs=None):
        print(f"Final learning rate: {self.model.optimizer.learning_rate.numpy()}")
        print(f"Final wait: {self.wait}")




def test_model(model, test_gen, data_min, data_max, filepath, tot=100):
    RMSE = []

    # Generate and evaluate tot batches
    for _ in range(tot):
        # Generate a batch
        x_true, y_true = next(test_gen)
        predictions = model.predict(x_true, verbose=0)

        # Denormalize
        predictions_denorm = ((predictions[..., 0] + 1) / 2) * (data_max - data_min) + data_min
        true_values_denorm = ((y_true[..., 0] + 1) / 2) * (data_max - data_min) + data_min

        # Get the masks and calculate the errors
        diffMask = y_true[..., 2]  # 1 only for hidden sea, 0 for land/clouds and visible sea

        diff_errors_batch = np.where(diffMask, np.abs(predictions_denorm - true_values_denorm), np.nan)

        # Calculate RMSE for each mask and append to respective lists
        squared_errors = np.nanmean(diff_errors_batch**2, axis=(1,2))
        RMSE.append(np.sqrt(squared_errors))

    # Concatenate RMSE values
    RMSE = np.concatenate(RMSE)
    rmse_mean = np.nanmean(RMSE)
    rmse_std = np.nanstd(RMSE)

    # Create the dictionary with the results to save
    result_dict = {
        'RMSE': rmse_mean,
        'RMSE_std': rmse_std
    }

    print(filepath)
    save_results(filepath, **result_dict)

    # Calculate and print the average RMSE for all error types
    print(f"RMSE all:", rmse_mean, ", std:", rmse_std)


def test_model_gaussian(model, test_gen, data_mean, data_std, filepath, tot=100):
    # Initialize lists to store the errors and the maximum errors
    RMSE = []

    # Generate and evaluate tot batches
    for _ in range(tot):
        # Generate a batch
        batch_x, batch_y = next(test_gen)
        #uncomment the next line and call your model
        predictions = model.predict(batch_x, verbose=0)[...,0]
        #predictions = batch_x[...,3] #use the baseline as prediction

        # Denormalize data !!!
        predictions_denorm = predictions*data_std + data_mean
        true_values_denorm = batch_y[..., 0]*data_std + data_mean

        # Remove the extra dimension from predictions_denorm using squeeze
        # predictions_denorm = np.squeeze(predictions_denorm)

        # Get the masks and calculate the errors
        diffMask = batch_y[..., 2]
        diff_errors_batch = np.where(diffMask, np.abs(predictions_denorm - true_values_denorm), np.nan)
        squared_errors = np.nanmean(diff_errors_batch**2,axis=(1,2))
        RMSE.append(np.sqrt(squared_errors))

    RMSE = np.concatenate(RMSE)
    rmse_mean = np.nanmean(RMSE)
    rmse_std = np.nanstd(RMSE)

    # Create the dictionary with the results to save
    result_dict = {
        'RMSE': rmse_mean,
        'RMSE_std': rmse_std
    }

    print(filepath)
    save_results(filepath, **result_dict)

    # Calculate and print the average RMSE for all error types
    print(f"(Gaussian norm) RMSE all:", rmse_mean, ", std:", rmse_std)

def test_dincae_gaussian(model, test_gen, data_mean, data_std, filepath, len_dataset, batch_size=32):
    # Initialize lists to store the errors and the maximum errors
    RMSE = []

    # Generate and evaluate tot batches
    for i in range(len_dataset//batch_size):
        # Generate a batch
        if i < (len_dataset // batch_size):
            #x_true, y_true, dates = next(dincae_test_gen)
            batch_x, batch_y = next(test_gen)
        else:
            # Special case for the last batch
            remaining_size = len_dataset % batch_size
            if remaining_size == 0:
                break
            #x_true, y_true, dates = next(dincae_test_gen)
            batch_x, batch_y = next(test_gen)
            batch_x = batch_x[:remaining_size]
            batch_y = batch_y[:remaining_size]

        predictions = model.predict(batch_x, verbose=0)
        
        # Denormalize data !!!
        predictions_denorm = predictions*data_std + data_mean
        true_values_denorm = batch_y[..., 0]*data_std + data_mean

        # Remove the extra dimension from predictions_denorm using squeeze
        predictions_denorm = np.squeeze(predictions_denorm)

        # Get the masks and calculate the errors
        diffMask = batch_y[..., 2]
        diff_errors_batch = np.where(diffMask, np.abs(predictions_denorm - true_values_denorm), np.nan)
        squared_errors = np.nanmean(diff_errors_batch**2,axis=(1,2))
        RMSE.append(np.sqrt(squared_errors))

    RMSE = np.concatenate(RMSE)
    rmse_mean = np.nanmean(RMSE)
    rmse_std = np.nanstd(RMSE)

    # Create the dictionary with the results to save
    result_dict = {
        'RMSE': rmse_mean,
        'RMSE_std': rmse_std
    }

    print(filepath)
    save_results(filepath, **result_dict)

    # Calculate and print the average RMSE for all error types
    print(f"(Gaussian norm) RMSE all:", rmse_mean, ", std:", rmse_std)


#%%
# Functions for saving results in npz files

def save_results(file, **kwargs):
    data_dict = {}
    # Check if the file exists and eventually load the data
    # If the file does not exist, create a new dictionary with the keys from kwargs
    try:
        data = np.load(file)
        data_dict = {key: value for key, value in data.items()}
    except (FileNotFoundError, EOFError):
        data_dict = {key: [] for key in kwargs}

    # Append the new values to the dictionary
    for key, value in kwargs.items():
        #print(f"key: {key}, value: {value}")
        if key in data_dict:
            data_dict[key] = np.append(data_dict[key], value)
        else:
            data_dict[key] = np.array([value])
    
    np.savez(file, **data_dict)
    print_results(file)
    
    
def clear_results(file):
    data = np.load(file)
    data_dict = {key: [] for key in data}
    np.savez(file, **data_dict)
    # print_results(file)


def clear_positional_results(file, position):
    # Given a position, remove the element at that position from all the arrays in the npz file
    data = np.load(file)
    data_dict = {key: value for key, value in data.items()}
    for key, value in data_dict.items():
        data_dict[key] = np.delete(data_dict[key], position)
    np.savez(file, **data_dict)
    # print_results(file)


def print_results(file):
    data = np.load(file)
    for key, value in data.items():
        print(f"{key}: {value}")




#%%
# Function for saving the history of the training
def save_history(history, path):
    metrics_to_save = ['loss', 'val_loss', 'RMSEMetric', 'val_RMSEMetric']
    try:
        data = np.load(path)
        data_dict = {key: value for key, value in data.items()}
    except (FileNotFoundError, EOFError):
        data_dict = {key: [] for key in metrics_to_save}

    # Append the new values to the dictionary
    for key in metrics_to_save:
        if key in data_dict:
            data_dict[key] = np.append(data_dict[key], history.history[key])
        else:
            data_dict[key] = np.array([history.history[key]])
    
    np.savez(path, **data_dict)
    # print_results(path)
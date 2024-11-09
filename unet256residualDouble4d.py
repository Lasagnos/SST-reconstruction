#Imports
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

from testUtils import TestCallbackGaussian

print('---------------------')

basepath_angelo = '/leonardo_scratch/fast/IscrC_AIWAF2/angelo/'
basepath_prof = '/leonardo_scratch/fast/IscrC_AIWAF2/andrea/'

# Load the residual dataset (only the night dataset)
residual_n = np.load(basepath_prof+'Angelo/datasets/residual_n.npy')
date_n = np.load(basepath_angelo+'datasets/date_n.npy')

# Load Italy mask, 256x256
italy_mask = np.load(basepath_angelo+'datasets/italy_mask.npy')

# Load baseline (not normalized)
abs_custom_baseline_n = np.load(basepath_prof+'Angelo/datasets/abs_new2_baseline_n.npy')
# turn in nans the values that are outside italy_mask (thus, it removes the zeros before calculating min and max)
baseline_n = np.where(italy_mask, abs_custom_baseline_n, np.nan)

print(f"residual_n: min: {np.nanmin(residual_n)}, max: {np.nanmax(residual_n)}")
print(f"baseline_n: min: {np.nanmin(baseline_n)}, max: {np.nanmax(baseline_n)}")

# Calculate the mean and the standard deviation of the residual dataset for normalization
residual_n_mean = np.nanmean(residual_n)
residual_n_std = np.nanstd(residual_n)
print(f"residual mean = {residual_n_mean}, residual_std = {residual_n_std}")

residual_n_norm = (residual_n - residual_n_mean) / residual_n_std

#DEBUG
print(f"residualN mean = {np.nanmean(residual_n_norm)}, residualN_std = {np.nanstd(residual_n_norm)}")  

# Split the day and the night dataset into training(16 years), validation(3 years), and testing(2 years) sets
def split_data(dataset):
    """Split dataset and dates into training, validation, and testing sets. Linear."""
    x_train = dataset[0:5832] # from 2002-07-04 to 2018-07-04
    x_val = dataset[5832:6922] # from 2018-07-04 to 2021-07-04
    x_test = dataset[6922:] # from 2021-07-04 to 2023-12-31

    return x_train, x_val, x_test

x_train_n, x_val_n, x_test_n = split_data(residual_n_norm)
dates_train_n, dates_val_n, dates_test_n = split_data(date_n)

# %%
def customLoss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    real_mask = y_true[:,:,:,1:2]        # 0 for land/clouds, 1 for sea
    y_true = y_true[:,:,:,0:1]          # The true SST values. Obfuscated areas are already converted to 0
    
    # Calculate the squared error only over clear sea
    squared_error = tf.square(y_true - y_pred)
    masked_error = squared_error * real_mask

    # Calculate the mean of the masked errors
    clear_loss = tf.reduce_mean(masked_error)     # The final loss

    return clear_loss

# %%
def ArtificialMetric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)    # Was getting an error because of the different types: y_true in the metrics is float64 instead of the normal float32

    diff_mask = y_true[:,:,:,2:3]  # 1 only for hidden sea, 0 for land/clouds and visible sea
    y_true = y_true[:,:,:,0:1]  # The true SST values. Obfuscated areas are already converted to 0

    # Calculate the squared error only over artificially clouded areas
    squared_error = tf.square(y_true - y_pred)
    artificial_masked_error = squared_error * diff_mask
    # Calculate the mean of the masked errors
    art_metric = tf.reduce_sum(artificial_masked_error) / tf.reduce_sum(diff_mask)

    return art_metric

#%%
# RMSE metrics
def RMSEMetric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)

    real_mask = y_true[:,:,:,1:2]  # 0 for land/clouds, 1 for clear sea
    art_mask = y_true[:,:,:,2:3]  # 0 for land/clouds + artificials, 1 for clear sea
    added_mask = real_mask - art_mask  # 1 only for hidden sea, 0 for land/clouds and visible sea
    y_true = y_true[:,:,:,0:1]  # The true SST values. Obfuscated areas are already converted to 0

    # Denormalize the predictions and the true values
    y_pred_denorm = y_pred * residual_n_std + residual_n_mean
    y_true_denorm = y_true * residual_n_std + residual_n_mean

    # Calculate the squared error only over hidden sea
    squared_error = tf.square(y_true_denorm - y_pred_denorm)
    hidden_masked_error = squared_error * added_mask
    # Calculate the mean of the masked errors
    mse_metric = tf.reduce_sum(hidden_masked_error) / tf.reduce_sum(added_mask)

    # Calculate the square root of the mean squared error to get the RMSE
    rmse_loss = tf.sqrt(mse_metric)

    return rmse_loss

# %%
#Generator hyperparameters
batch_size = 32

isize = 256
previous_days = 4   # How many previous days to consider in the input
n_channels = 3 + previous_days*2   # 2 for the current day, 1 for basic info and 2 for each previous day
input_shape = (isize, isize, n_channels)

# %%
# Generator for the residual 4 days model
# Creation of the baseline not commented, but it is not used in the model
def generator(batch_size, dataset, date):
    while True:
        batch_x = np.zeros((batch_size, isize, isize, n_channels))
        batch_y = np.zeros((batch_size, isize, isize, 3))

        for b in range(batch_size):
            # Choose a random index as the current day
            found = False
            while not found:
              i = np.random.randint(4, dataset.shape[0])    # Start from 4 to have at least 4 previous days
              # Ensure that the starting image has at least 40% coverage
              if np.sum(~np.isnan(dataset[i])/(isize*isize)) > 0.4:
                found = True

            image_current = np.nan_to_num(dataset[i], nan=0)
            mask_current = np.isnan(dataset[i])
            image_prev_1 = np.nan_to_num(dataset[i-1], nan=0)
            mask_prev_1 = np.isnan(dataset[i-1])
            image_prev_2 = np.nan_to_num(dataset[i-2], nan=0)
            mask_prev_2 = np.isnan(dataset[i-2])
            image_prev_3 = np.nan_to_num(dataset[i-3], nan=0)
            mask_prev_3 = np.isnan(dataset[i-3])
            image_prev_4 = np.nan_to_num(dataset[i-4], nan=0)
            mask_prev_4 = np.isnan(dataset[i-4])


            # Extending clouds
            found = False
            while not found:
              r = np.random.randint(4, dataset.shape[0])    # Start from 4 to have at least 4 previous days
              mask_r = np.isnan(dataset[r])
              mask_or_r = np.logical_or(mask_current, mask_r)
              # Ensure that the image has between 60% and 85% coverage
              nans = np.sum(mask_or_r)/(isize*isize)
              if nans > 0.6 and nans < 0.85:
                found = True

            # Fix artificial mask
            artificial_mask_current = np.logical_not(mask_or_r)  #1 visible, 0 masked
            # we try to keep the coerence of the temporal evolution of the clouds
            artificial_mask_prev_1 = ~np.logical_or(mask_prev_1, np.isnan(dataset[r-1]))
            artificial_mask_prev_2 = ~np.logical_or(mask_prev_2, np.isnan(dataset[r-2]))
            artificial_mask_prev_3 = ~np.logical_or(mask_prev_3, np.isnan(dataset[r-3]))
            artificial_mask_prev_4 = ~np.logical_or(mask_prev_4, np.isnan(dataset[r-4]))

            # Apply the amplified mask to the current day's image
            image_masked_current = np.where(artificial_mask_current, image_current, 0)
            image_masked_prev_1 = np.where(artificial_mask_prev_1, image_prev_1, 0)
            image_masked_prev_2 = np.where(artificial_mask_prev_2, image_prev_2, 0)
            image_masked_prev_3 = np.where(artificial_mask_prev_3, image_prev_3, 0)
            image_masked_prev_4 = np.where(artificial_mask_prev_4, image_prev_4, 0)

            # Convert the current date to a datetime object using pandas
            date_series = pd.to_datetime(date[i], unit='D', origin='unix')
            day_of_year = date_series.dayofyear

            # Tune the baseline to match the average temperature of the current day (UNUSED!!!)
            image_masked_nan = np.where(artificial_mask_current, image_current, np.nan)
            avg_temp_real = np.nanmean(image_masked_nan)
            avg_temp_baseline = np.nanmean(np.where(artificial_mask_current, baseline_n[day_of_year - 1], np.nan))
            tuned_baseline = baseline_n[day_of_year - 1] + avg_temp_real - avg_temp_baseline  # Adjust the baseline to match the average temperature of the current day
            tuned_baseline = np.where(italy_mask, tuned_baseline, 0)    # Apply the land-sea mask

            # Fix masks before they are used in the loss and metric functions
            mask_current = np.logical_not(mask_current) # 1 for clear sea, 0 for land/clouds
            diff_mask = np.logical_and(~artificial_mask_current, mask_current)

            # Create batch_x and batch_y
            batch_x[b, ..., 0] = image_masked_current       #artificially cloudy image
            batch_x[b, ..., 1] = artificial_mask_current    #artificial mask
            batch_x[b, ..., 2] = image_masked_prev_1          #prev1 artificially clouded image
            batch_x[b, ..., 3] = artificial_mask_prev_1       #prev1 artificial mask
            batch_x[b, ..., 4] = image_masked_prev_2          #prev2 artificially clouded image
            batch_x[b, ..., 5] = artificial_mask_prev_2       #prev2 artificial mask
            batch_x[b, ..., 6] = image_masked_prev_3          #prev3 artificially clouded image
            batch_x[b, ..., 7] = artificial_mask_prev_3       #prev3 artificial mask
            batch_x[b, ..., 8] = image_masked_prev_4          #prev4 artificially clouded image
            batch_x[b, ..., 9] = artificial_mask_prev_4       #prev4 artificial mask
            batch_x[b, ..., 10] = italy_mask                 #land-sea mask
            #batch_x[b, ..., 7] = baseline[day_of_year -1]   #tuned_baseline             #tuned baseline

            batch_y[b, ..., 0] = image_current              #real image
            batch_y[b, ..., 1] = mask_current               #real mask
            batch_y[b, ..., 2] = diff_mask                  #'hidden' mask

        yield batch_x, batch_y

train_gen = generator(batch_size, x_train_n, dates_train_n)
val_gen = generator(batch_size, x_val_n, dates_val_n)
test_gen = generator(batch_size, x_test_n, dates_test_n)


# %%
# Hyperparameters
epochs=200

# 0, 50, 100, 150
#lr = [1e-4, 0.75e-4, 0.5e-5, 0.3e-4]
lr = 1e-4   # starting learning rate

loss = customLoss
metrics = [ArtificialMetric, RMSEMetric]

steps_per_epoch = len(x_train_n) // batch_size
validation_steps = len(x_val_n) // batch_size
testing_steps = len(x_test_n) // batch_size


# %%
# Define the U-Net model
def ResidualBlock(depth):
    def apply(x):
        input_depth = x.shape[3]    # Get the number of channels from the channels dimension
        if input_depth == depth:    # It's already the desired channel number
            residual = x
        else:                       # Adjust the number of channels with a 1x1 convolution
            residual = Conv2D(depth, kernel_size=1)(x)

        x = BatchNormalization(center=False, scale=False)(x)
        x = Conv2D(depth, kernel_size=3, padding="same", activation='swish')(x)
        x = Conv2D(depth, kernel_size=3, padding="same")(x)
        x = Add()([x, residual])
        return x

    return apply


def DownBlock(depth, block_depth):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(depth)(x)
            skips.append(x)
        x = AveragePooling2D(pool_size=2)(x)    #downsampling
        return x

    return apply


def UpBlock(depth, block_depth):
    def apply(x):
        x, skips = x
        x = UpSampling2D(size=2, interpolation="bilinear")(x)   #upsampling
        for _ in range(block_depth):
            x = Concatenate()([x, skips.pop()])
            x = ResidualBlock(depth)(x)
        return x

    return apply


def get_Unet(image_size, depths, block_depth):
    input_images = Input(shape=image_size)  #input layer

    x = Conv2D(depths[0], kernel_size=1)(input_images)  #reduce the number of channels

    skips = []  #store the skip connections

    for depth in depths[:-1]:   #downsampling layers
        x = DownBlock(depth, block_depth)([x, skips])

    for _ in range(block_depth):    #middle layer
        x = ResidualBlock(depths[-1])(x)

    for depth in reversed(depths[:-1]):   #upsampling layers
        x = UpBlock(depth, block_depth)([x, skips])

    x = Conv2D(1, kernel_size=1, kernel_initializer="zeros", name = "output_noise")(x)  #no activation function

    return Model(input_images, outputs=x, name="UNetInpainter")

# Define the model
#depths = [32, 64, 128, 256]         #HYPERPARAMETER
depths = [64, 128, 256, 512]         #HYPERPARAMETER
block_depth = 2                     #HYPERPARAMETER
model = get_Unet(input_shape, depths, block_depth)

# Compile the model with custom loss and Adam optimizer
opt = Adam(learning_rate=lr)
model.compile(optimizer=opt, loss=loss, metrics=metrics)
#model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-5, min_delta=1e-3)
model_checkpoint = ModelCheckpoint(filepath='weights/unet256residualDouble4d.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, verbose=1) # min_delta=1e-3
test_callbacks = TestCallbackGaussian(test_gen, residual_n_mean, residual_n_std, 'results/callbackUnet256resDouble4d.npz', 10)
callbacks = [early_stop, reduce_lr, model_checkpoint, test_callbacks]

# LOAD WEIGHTS
#model.load_weights('weights/unet256residualDouble4d.h5')

# TRAIN MODEL (if True)
if True:
    # Train model
    start_time = time.time()
    history = model.fit(train_gen, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=val_gen, validation_steps=validation_steps, verbose=2, callbacks=callbacks)
    end_time = time.time()
    #SAVE WEIGHTS
    #model.save_weights('weights/unet256residualDouble4d.h5')  # saved by the checkpoint callback

    total_time = end_time - start_time
    print(f"\nTotal time spent training on {epochs} epochs: {total_time} seconds")


# Initialize lists to store the errors and the maximum errors
RMSE = []

# Generate and evaluate tot batches
tot = 50
for _ in range(tot):
    # Generate a batch
    batch_x, batch_y = next(test_gen)
    #uncomment the next line and call your model
    predictions = model.predict(batch_x, verbose=0)[...,0]
    #predictions = batch_x[...,3] #use the baseline as prediction

    # Denormalize data !!!
    predictions_denorm = predictions*residual_n_std + residual_n_mean
    true_values_denorm = batch_y[..., 0]*residual_n_std + residual_n_mean

    # Remove the extra dimension from predictions_denorm using squeeze
    # predictions_denorm = np.squeeze(predictions_denorm)

    # Get the masks and calculate the errors
    diffMask = batch_y[..., 2]
    diff_errors_batch = np.where(diffMask, np.abs(predictions_denorm - true_values_denorm), np.nan)
    squared_errors = np.nanmean(diff_errors_batch**2,axis=(1,2))
    RMSE.append(np.sqrt(squared_errors))

RMSE = np.concatenate(RMSE)

print(f"RMSE :", np.mean(RMSE))
print(f"RMSE std :", np.std(RMSE))

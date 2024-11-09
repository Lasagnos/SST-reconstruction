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

from sklearn.model_selection import train_test_split

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
abs_new2_baseline_n = np.load(basepath_prof+'Angelo/datasets/abs_new2_baseline_n.npy')
# turn in nans the values that are outside italy_mask (thus, it removes the zeros before calculating min and max)
abs_baseline_n = np.where(italy_mask, abs_new2_baseline_n, np.nan)

print(f"residual_n: min: {np.nanmin(residual_n)}, max: {np.nanmax(residual_n)}")
print(f"baseline_n: min: {np.nanmin(abs_baseline_n)}, max: {np.nanmax(abs_baseline_n)}")

# Calculate the mean and the standard deviation of the residual dataset for normalization
residual_n_mean = np.nanmean(residual_n)
residual_n_std = np.nanstd(residual_n)
print(f"residual mean = {residual_n_mean}, residual_std = {residual_n_std}")

residual_n_norm = (residual_n - residual_n_mean) / residual_n_std


is128 = True   ############################# 128x128 or 256x256. For data initialization and splitting
isDouble = True    ############################# Double or normal channels. [64, 128, 256, 512] or [32, 64, 128, 256]
previous_days = 5   ############################# How many previous days to consider in the input

weights_path = 'weights/unet256residualDouble5d.h5'                 # for 256x256
weights_path_128_NW = 'weights/unet128residualDouble_NW_5d.h5'      # for 128x128 quadrants
weights_path_128_NE = 'weights/unet128residualDouble_NE_5d.h5'
weights_path_128_SW = 'weights/unet128residualDouble_SW_5d.h5'
weights_path_128_SE = 'weights/unet128residualDouble_SE_5d.h5'


if is128:
    # %%
    # Split the datasets into 4 parts of size 128x128

    def split_dataset(dataset, size=128):
        """Split a dataset into four parts"""

        # Resize each part
        nw = dataset[:, :size, :size]
        ne = dataset[:, :size, size:]
        sw = dataset[:, size:, :size]
        se = dataset[:, size:, size:]

        return nw, ne, sw, se

    def split_mask(mask, size=128):
        """Split a mask into four parts."""

        # Resize each part
        nw = mask[:size, :size]
        ne = mask[:size, size:]
        sw = mask[size:, :size]
        se = mask[size:, size:]

        return nw, ne, sw, se

    dataset_n_NW, dataset_n_NE, dataset_n_SW, dataset_n_SE = split_dataset(residual_n_norm)
    baseline_n_NW, baseline_n_NE, baseline_n_SW, baseline_n_SE = split_dataset(abs_baseline_n)
    italy_mask_NW, italy_mask_NE, italy_mask_SW, italy_mask_SE = split_mask(italy_mask)

    # Dictionary to map masks and baselines to the corresponding quadrant
    quadrant_map = {
        "NW": {
            "italy_mask": italy_mask_NW,
            "baseline_n": baseline_n_NW
        },
        "NE": {
            "italy_mask": italy_mask_NE,
            "baseline_n": baseline_n_NE
        },
        "SW": {
            "italy_mask": italy_mask_SW,
            "baseline_n": baseline_n_SW
        },
        "SE": {
            "italy_mask": italy_mask_SE,
            "baseline_n": baseline_n_SE
        }
    }

    # Split the day and the night dataset into training(16 years), validation(3 years), and testing(2 years) sets
    def split_data(dataset):
        """Split dataset and dates into training, validation, and testing sets. Linear."""
        x_train = dataset[0:5832] # from 2002-07-04 to 2018-07-04
        x_val = dataset[5832:6922] # from 2018-07-04 to 2021-07-04
        x_test = dataset[6922:] # from 2021-07-04 to 2023-12-31

        return x_train, x_val, x_test

    x_train_n_NW, x_val_n_NW, x_test_n_NW = split_data(dataset_n_NW)
    x_train_n_NE, x_val_n_NE, x_test_n_NE = split_data(dataset_n_NE)
    x_train_n_SW, x_val_n_SW, x_test_n_SW = split_data(dataset_n_SW)
    x_train_n_SE, x_val_n_SE, x_test_n_SE = split_data(dataset_n_SE)
    dates_train_n, dates_val_n, dates_test_n = split_data(date_n)
else:
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

isize = 128 if is128 else 256
#previous_days = 2   # How many previous days to consider in the input
n_channels = 3 + previous_days*2   # 2 for the current day 2 for each previous day and 1 for land-sea mask
input_shape = (isize, isize, n_channels)

# %%
# MODULAR GENERATOR
def generator(batch_size, dataset, date, quadrant="O"): #O for the whole dataset
    while True:
        batch_x = np.zeros((batch_size, isize, isize, n_channels))
        batch_y = np.zeros((batch_size, isize, isize, 3))

        if is128:
            if quadrant in quadrant_map:    # Get the mask and baseline for the corresponding quadrant
                land_sea_mask = quadrant_map[quadrant]["italy_mask"]
                #baseline_n = quadrant_map[quadrant]["baseline_n"]  # UNUSED!!!
        else:
            land_sea_mask = italy_mask
            #baseline_n = abs_baseline_n  # UNUSED!!!

        for b in range(batch_size):
            # Choose a random index as the current day
            found = False
            while not found:
              i = np.random.randint(previous_days, dataset.shape[0])    # Start from 4 to have at least 4 previous days
              # Ensure that the starting image has at least 40% coverage
              if np.sum(~np.isnan(dataset[i])/(isize*isize)) > 0.4:
                found = True

            image_current = np.nan_to_num(dataset[i], nan=0)
            mask_current = np.isnan(dataset[i])

            # Extending clouds
            found = False
            while not found:
              r = np.random.randint(previous_days, dataset.shape[0])    # Start from 4 to have at least 4 previous days
              mask_r = np.isnan(dataset[r])
              mask_or_r = np.logical_or(mask_current, mask_r)
              # Ensure that the image has between 60% and 85% coverage
              nans = np.sum(mask_or_r)/(isize*isize)
              if nans > 0.6 and nans < 0.85:
                found = True

            # Fix artificial mask
            artificial_mask_current = np.logical_not(mask_or_r)  #1 visible, 0 masked
            # Apply the amplified mask to the current day's image
            image_masked_current = np.where(artificial_mask_current, image_current, 0)

            # Get the previous days' images and masks
            images_prev = []
            masks_prev = []
            artificial_masks_prev = []
            images_masked_prev = []
            for j in range(previous_days):
                image_prev = np.nan_to_num(dataset[i-j-1], nan=0)
                mask_prev = np.isnan(dataset[i-j-1])
                artificia_mask_prev = ~np.logical_or(mask_prev, np.isnan(dataset[r-j-1]))
                image_masked_prev = np.where(artificia_mask_prev, image_prev, 0)

                images_prev.append(image_prev)
                masks_prev.append(mask_prev)
                artificial_masks_prev.append(artificia_mask_prev)
                images_masked_prev.append(image_masked_prev)


            # # Convert the current date to a datetime object using pandas  (UNUSED!!!)
            # date_series = pd.to_datetime(date[i], unit='D', origin='unix')
            # day_of_year = date_series.dayofyear

            # # Tune the baseline to match the average temperature of the current day (UNUSED!!!)
            # image_masked_nan = np.where(artificial_mask_current, image_current, np.nan)
            # avg_temp_real = np.nanmean(image_masked_nan)
            # avg_temp_baseline = np.nanmean(np.where(artificial_mask_current, baseline_n[day_of_year - 1], np.nan))
            # tuned_baseline = baseline_n[day_of_year - 1] + avg_temp_real - avg_temp_baseline  # Adjust the baseline to match the average temperature of the current day
            # tuned_baseline = np.where(italy_mask, tuned_baseline, 0)    # Apply the land-sea mask

            # Fix masks before they are used in the loss and metric functions
            mask_current = np.logical_not(mask_current) # 1 for clear sea, 0 for land/clouds
            diff_mask = np.logical_and(~artificial_mask_current, mask_current)

            # Create batch_x and batch_y
            batch_x[b, ..., 0] = image_masked_current       #artificially cloudy image
            batch_x[b, ..., 1] = artificial_mask_current    #artificial mask
            batch_x[b, ..., n_channels-1] = land_sea_mask   #land-sea mask (italy_mask)
            #batch_x[b, ..., n_channels-1] = baseline[day_of_year -1]   #tuned_baseline             #tuned baseline
            for j in range(previous_days):
                batch_x[b, ..., (j*2)+2] = images_masked_prev[j]
                batch_x[b, ..., (j*2)+3] = artificial_masks_prev[j]


            batch_y[b, ..., 0] = image_current              #real image
            batch_y[b, ..., 1] = mask_current               #real mask
            batch_y[b, ..., 2] = diff_mask                  #'hidden' mask
            #batch_y[b, ..., 3] = baseline[day_of_year - 1]  #baseline

        yield batch_x, batch_y

if is128:
    train_gen_NW = generator(batch_size, x_train_n_NW, dates_train_n, "NW")
    train_gen_NE = generator(batch_size, x_train_n_NE, dates_train_n, "NE")
    train_gen_SW = generator(batch_size, x_train_n_SW, dates_train_n, "SW")
    train_gen_SE = generator(batch_size, x_train_n_SE, dates_train_n, "SE")
    val_gen_NW = generator(batch_size, x_val_n_NW, dates_val_n, "NW")
    val_gen_NE = generator(batch_size, x_val_n_NE, dates_val_n, "NE")
    val_gen_SW = generator(batch_size, x_val_n_SW, dates_val_n, "SW")
    val_gen_SE = generator(batch_size, x_val_n_SE, dates_val_n, "SE")
    test_gen_NW = generator(batch_size, x_test_n_NW, dates_test_n, "NW")
    test_gen_NE = generator(batch_size, x_test_n_NE, dates_test_n, "NE")
    test_gen_SW = generator(batch_size, x_test_n_SW, dates_test_n, "SW")
    test_gen_SE = generator(batch_size, x_test_n_SE, dates_test_n, "SE")
else:
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

if is128:
    steps_per_epoch = len(x_train_n_NW) // batch_size
    validation_steps = len(x_val_n_NW) // batch_size
    testing_steps = len(x_test_n_NW) // batch_size
else:
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
depths = [64, 128, 256, 512] if isDouble else [32, 64, 128, 256]     #HYPERPARAMETER
block_depth = 2                     #HYPERPARAMETER
if is128:
    model_NW = get_Unet(input_shape, depths, block_depth)
    model_NE = get_Unet(input_shape, depths, block_depth)
    model_SW = get_Unet(input_shape, depths, block_depth)
    model_SE = get_Unet(input_shape, depths, block_depth)
else:
    model = get_Unet(input_shape, depths, block_depth)

# Compile the model with custom loss and Adam optimizer
if is128:
    optNW = Adam(learning_rate=lr)  # like this, every model will have an equal starting learning rate
    model_NW.compile(optimizer=optNW, loss=loss, metrics=[metrics])
    optNE = Adam(learning_rate=lr)
    model_NE.compile(optimizer=optNE, loss=loss, metrics=[metrics])
    optSW = Adam(learning_rate=lr)
    model_SW.compile(optimizer=optSW, loss=loss, metrics=[metrics])
    optSE = Adam(learning_rate=lr)
    model_SE.compile(optimizer=optSE, loss=loss, metrics=[metrics])
else:
    opt = Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
#model.summary()


# LOAD WEIGHTS
if is128:
    model_NW.load_weights(weights_path_128_NW)
    model_NE.load_weights(weights_path_128_NE)
    model_SW.load_weights(weights_path_128_SW)
    model_SE.load_weights(weights_path_128_SE)
else:
    model.load_weights(weights_path)


# %%
print('---------------------')
# VALIDATION

if is128:
    # Initialize lists to store the errors and the maximum errors for each quadrant
    all_RMSEs = {"NW": [], "NE": [], "SW": [], "SE": []}
    all_batch_stds = {"NW": [], "NE": [], "SW": [], "SE": []}
    # Define the quadrants
    quadrants = ["NW", "NE", "SW", "SE"]
    # Repeat the validation 10 times for each quadrant
    num_repeats = 10
    tot = 50

    for quadrant in quadrants:
        print(f"Evaluating quadrant: {quadrant}")

        for repeat in range(num_repeats):
            RMSE = []

            # Select the appropriate test generator and model for the current quadrant
            if quadrant == "NW":
                test_gen = test_gen_NW
                model = model_NW
            elif quadrant == "NE":
                test_gen = test_gen_NE
                model = model_NE
            elif quadrant == "SW":
                test_gen = test_gen_SW
                model = model_SW
            elif quadrant == "SE":
                test_gen = test_gen_SE
                model = model_SE

            for _ in range(tot):
                # Generate a batch
                batch_x, batch_y = next(test_gen)
                # Uncomment the next line and call your model
                predictions = model.predict(batch_x, verbose=0)[..., 0]
                # predictions = batch_x[..., 3] # Use the baseline as prediction

                # Denormalize data
                predictions_denorm = predictions * residual_n_std + residual_n_mean
                true_values_denorm = batch_y[..., 0] * residual_n_std + residual_n_mean

                # Get the masks and calculate the errors
                diffMask = batch_y[..., 2]
                diff_errors_batch = np.where(diffMask, np.abs(predictions_denorm - true_values_denorm), np.nan)
                squared_errors = np.nanmean(diff_errors_batch**2, axis=(1, 2))
                RMSE.append(np.sqrt(squared_errors))

            RMSE = np.concatenate(RMSE)
            all_RMSEs[quadrant].append(np.nanmean(RMSE))
            all_batch_stds[quadrant].append(np.nanstd(RMSE))

    # Calculate the final RMSE and standard deviation for each quadrant
    final_results = {}
    for quadrant in quadrants:
        final_RMSE = np.nanmean(all_RMSEs[quadrant])
        final_std = np.nanstd(all_RMSEs[quadrant])
        final_results[quadrant] = (final_RMSE, final_std)
        print(f"Quadrant {quadrant} - Array of RMSEs: {all_RMSEs[quadrant]}")
        print(f"Quadrant {quadrant} - Array of batch standard deviations: {all_batch_stds[quadrant]}")
        print(f"Quadrant {quadrant} - Final result: RMSE = {final_RMSE} ± {final_std}")

    # Calculate the average final RMSE and standard deviation across all quadrants
    average_final_RMSE = np.nanmean([result[0] for result in final_results.values()])
    average_final_std = np.nanstd([result[0] for result in final_results.values()])

    print(f"Average final RMSE across all quadrants: {average_final_RMSE} ± {average_final_std}")
else:
    # Initialize lists to store the errors and the maximum errors
    all_RMSEs = []
    all_batch_stds = []

    # Repeat the validation 10 times
    num_repeats = 10
    for repeat in range(num_repeats):
        RMSE = []

        # Generate and evaluate tot batches
        tot = 50
        for _ in range(tot):
            # Generate a batch
            batch_x, batch_y = next(test_gen)
            predictions = model.predict(batch_x, verbose=0)[...,0]

            # Denormalize data !!!
            predictions_denorm = predictions*residual_n_std + residual_n_mean
            true_values_denorm = batch_y[..., 0]*residual_n_std + residual_n_mean

            # Get the masks and calculate the errors
            diffMask = batch_y[..., 2]
            diff_errors_batch = np.where(diffMask, np.abs(predictions_denorm - true_values_denorm), np.nan)
            squared_errors = np.nanmean(diff_errors_batch**2,axis=(1,2))
            RMSE.append(np.sqrt(squared_errors))

        RMSE = np.concatenate(RMSE)
        all_RMSEs.append(np.nanmean(RMSE))
        all_batch_stds.append(np.nanstd(RMSE))

    # Calculate the final RMSE and standard deviation
    final_RMSE = np.nanmean(all_RMSEs)
    final_std = np.nanstd(all_RMSEs)

    # Print the results
    print(f"Array of RMSEs: {all_RMSEs}")
    print(f"Array of batch standard deviations: {all_batch_stds}")
    print(f"Final result: RMSE = {final_RMSE} ± {final_std}")

print('---------------------')
###################################################

# %%
# Dincae dataset (not the residual one)
dincae_date = np.arange(12053, 17167)   # from 2003-01-01 to 2016-12-31
msr_sst_t_array = np.load(basepath_angelo+'datasets/dincae/msr_sst_t_array.npy')    # No added cloud data
msrc_sst_t_array = np.load(basepath_angelo+'datasets/dincae/msrc_sst_t_array.npy')  # Added cloud data

# Split the data into training(70%), validation(20%), and testing(10%) sets
def split_data_dincae(dataset):
    """Split dataset and dates into training, validation, and testing sets. Linear."""
    x_train, x_temp = train_test_split(dataset, test_size=0.3, shuffle=False)
    x_val, x_test = train_test_split(x_temp, test_size=1/3, shuffle=False)

    return x_train, x_val, x_test

#WARNING  normalize data!
x_msr_train, x_msr_val, x_msr_test = split_data_dincae(msr_sst_t_array)
x_msrc_train, x_msrc_val, x_msrc_test = split_data_dincae(msrc_sst_t_array)
dates_train, dates_val, dates_test = split_data_dincae(dincae_date)

if is128:   # 128x128
    offset_y = 0
    offset_x = 30 # 30 pixels are cut off from the left
    end_y = 128   # since y is 144 pixels tall but it has to stop at 128, 16 pixels are cut off from the bottom
    end_x = 158   # since x is 168 pixels wide but it has to stop at 128, 30 pixels are cut off from the left and 10 from the right (the right 10 are outside even the 256x256 image)
    
    def dincae_generator(batch_size, msrData, msrcData, date, italy_mask, abs_baseline_n):
        i = previous_days
        while True:
            batch_x = np.zeros((batch_size, isize, isize, n_channels))
            batch_y = np.zeros((batch_size, isize, isize, 3))

            for b in range(batch_size):
                images = []
                masks = []
                artificial_masks = []
                image_masked = []
                for j in range(0, previous_days+1):
                    date_series = pd.to_datetime(date[i-j], unit='D', origin='unix')
                    day_of_year = date_series.dayofyear

                    # Prepare the images and masks
                    image = msrData[i-j]
                    image = image[offset_y:end_y, offset_x:end_x]   # (Correctly) cut the image to 128x128
                    image = image - abs_baseline_n[day_of_year-1]
                    image = (image - residual_n_mean) / residual_n_std
                    image = np.nan_to_num(image, nan=0)

                    # Prepare the masks
                    msr_mask = np.isnan(msrData[i-j])
                    msr_mask = msr_mask[offset_y:end_y, offset_x:end_x]

                    # Prepare the artificial masks
                    msrc_mask = np.isnan(msrcData[i-j])
                    msrc_mask = msrc_mask[offset_y:end_y, offset_x:end_x]

                    # if 256x256 the image have to be enlarged
                    # image, msr_mask, msrc_mask = enlarge_masks_and_images(image, msr_mask, msrc_mask)

                    # Append the images and masks
                    images.append(image)
                    masks.append(msr_mask)
                    artificial_masks.append(np.logical_not(msrc_mask))
                    image_masked.append(np.where(artificial_masks[j], images[j], 0))


                # Fix masks before they are used in the loss and metric functions
                mask_current = np.logical_not(masks[0]) # 1 for clear sea, 0 for land/clouds
                diff_mask = np.logical_and(~artificial_masks[0], mask_current)

            # Create batch_x and batch_y
                for j in range(0, previous_days+1):
                    batch_x[b, ..., j*2] = image_masked[j]                    #artificially cloudy image
                    batch_x[b, ..., j*2+1] = artificial_masks[j]              #artificial mask
                batch_x[b, ..., n_channels-1] = italy_mask                    #land-sea mask

                batch_y[b, ..., 0] = images[0]                                #real image
                batch_y[b, ..., 1] = mask_current                             #real mask
                batch_y[b, ..., 2] = diff_mask                                #artificial mask (for the metrics)

                # Increment the index
                i += 1
                if i >= msrData.shape[0]:
                    i = previous_days

            yield batch_x, batch_y

    dincae_gen = dincae_generator(batch_size, msr_sst_t_array, msrc_sst_t_array, dincae_date, italy_mask_NE, baseline_n_NE)
    dincae_gen_10 = dincae_generator(batch_size, x_msr_test, x_msrc_test, dates_test, italy_mask_NE, baseline_n_NE)
else:   # 256x256
    offset_y = 0
    offset_x = 0  
    end_y = 144   # since y is 144 pixels tall, all is used
    end_x = 158   # the right 10 are outside

    def enlarge_masks_and_images(msr_image, msr_mask, msrc_mask):
        # Create a 256x256 image and masks, and insert them in the right place
        # considering that our data is 256x256, and we overlap perfectly with the height but not with the width...
        # ...we have to pad the width by 256-158=98 pixels horizontally. The last 10 pixels are already cut off.
        greater_msr_image = np.zeros((256, 256), dtype=float)
        greater_msr_image[:144, 98:] = msr_image
        greater_msr_mask = np.ones((256, 256), dtype=bool)
        greater_msr_mask[:144, 98:] = msr_mask
        greater_msrc_mask = np.ones((256, 256), dtype=bool)
        greater_msrc_mask[:144, 98:] = msrc_mask
        return greater_msr_image, greater_msr_mask, greater_msrc_mask

    #GENERATOR
    def dincae_generator(batch_size, msrData, msrcData, date, italy_mask, abs_baseline_n):
        i = previous_days
        while True:
            batch_x = np.zeros((batch_size, isize, isize, n_channels))
            batch_y = np.zeros((batch_size, isize, isize, 3))

            for b in range(batch_size):
                images = []
                masks = []
                artificial_masks = []
                image_masked = []
                for j in range(0, previous_days+1):
                    date_series = pd.to_datetime(date[i-j], unit='D', origin='unix')
                    day_of_year = date_series.dayofyear

                    # Prepare the images and masks
                    image = msrData[i-j]
                    image = image[offset_y:end_y, offset_x:end_x]   # Get a 144x158 image
                    image = image - abs_baseline_n[day_of_year-1, :144, 98:]   # Subtract the offset baseline
                    image = (image - residual_n_mean) / residual_n_std
                    image = np.nan_to_num(image, nan=0)

                    # Prepare the masks
                    msr_mask = np.isnan(msrData[i-j])
                    msr_mask = msr_mask[offset_y:end_y, offset_x:end_x]

                    # Prepare the artificial masks
                    msrc_mask = np.isnan(msrcData[i-j])
                    msrc_mask = msrc_mask[offset_y:end_y, offset_x:end_x]

                    # if 256x256 the image have to be enlarged
                    image, msr_mask, msrc_mask = enlarge_masks_and_images(image, msr_mask, msrc_mask)

                    # Append the images and masks
                    images.append(image)
                    masks.append(msr_mask)
                    artificial_masks.append(np.logical_not(msrc_mask))
                    image_masked.append(np.where(artificial_masks[j], images[j], 0))


                # Fix masks before they are used in the loss and metric functions
                mask_current = np.logical_not(masks[0]) # 1 for clear sea, 0 for land/clouds
                diff_mask = np.logical_and(~artificial_masks[0], mask_current)

            # Create batch_x and batch_y
                for j in range(0, previous_days+1):
                    batch_x[b, ..., j*2] = image_masked[j]                    #artificially cloudy image
                    batch_x[b, ..., j*2+1] = artificial_masks[j]              #artificial mask
                batch_x[b, ..., n_channels-1] = italy_mask                    #land-sea mask

                batch_y[b, ..., 0] = images[0]                                #real image
                batch_y[b, ..., 1] = mask_current                             #real mask
                batch_y[b, ..., 2] = diff_mask                                #artificial mask (for the metrics)

                # Increment the index
                i += 1
                if i >= msrData.shape[0]:
                    i = previous_days

            yield batch_x, batch_y

    dincae_gen = dincae_generator(batch_size, msr_sst_t_array, msrc_sst_t_array, dincae_date, italy_mask, abs_baseline_n)
    dincae_gen_10 = dincae_generator(batch_size, x_msr_test, x_msrc_test, dates_test, italy_mask, abs_baseline_n)


# %%
# VALIDATION of dincae dataset
# Initialize lists to store the errors and the maximum errors
all_RMSEs = []
all_batch_stds = []

# Repeat the validation 10 times
num_repeats = 10
for _ in range(num_repeats):
    RMSE = []

    # Generate and evaluate all batches in the dataset
    for i in range((len(msr_sst_t_array)//batch_size)+1):
        # Generate a batch
        if i < len(msr_sst_t_array) // batch_size:
            batch_x, batch_y = next(dincae_gen)
        else:
            # Special case for the last batch
            remaining_size = len(msr_sst_t_array) % batch_size
            if remaining_size == 0:
                break
            batch_x, batch_y = next(dincae_gen)
            batch_x = batch_x[:remaining_size]
            batch_y = batch_y[:remaining_size]

        if is128:
            predictions = model_NW.predict(batch_x, verbose=0)[...,0]
        else:
            predictions = model.predict(batch_x, verbose=0)[...,0]

        # Denormalize data !!!
        predictions_denorm = predictions*residual_n_std + residual_n_mean
        true_values_denorm = batch_y[..., 0]*residual_n_std + residual_n_mean

        # Get the masks and calculate the errors
        diffMask = batch_y[..., 2]     #diff mask
        diff_errors_batch = np.where(diffMask, np.abs(predictions_denorm - true_values_denorm), np.nan)
        squared_errors = np.nanmean(diff_errors_batch**2, axis=(1, 2))# if np.any(~np.isnan(diff_errors_batch)) else np.nan
        #squared_errors = np.where(np.sum(diffMask,axis=(1,2))<100,np.nan,squared_errors)
        RMSE.append(np.sqrt(squared_errors))

    RMSE = np.concatenate(RMSE)
    all_RMSEs.append(np.nanmean(RMSE))
    all_batch_stds.append(np.nanstd(RMSE))

# Calculate the final RMSE and standard deviation
final_RMSE = np.nanmean(all_RMSEs)
final_std = np.nanstd(all_RMSEs)

# Print the results
print(f"Dincae - Array of RMSEs: {all_RMSEs}")
print(f"Dincae - Array of batch standard deviations: {all_batch_stds}")
print(f"Dincae - Final result: RMSE = {final_RMSE} ± {final_std}")


# %%
# VALIDATION of dincae dataset (last 10%)
print('---------------------')
# Initialize lists to store the errors and the maximum errors
all_RMSEs = []
all_batch_stds = []

# Repeat the validation 10 times
num_repeats = 10
for _ in range(num_repeats):
    RMSE = []

    # Generate and evaluate all batches in the dataset
    for i in range((len(x_msr_test)//batch_size)+1):
        # Generate a batch
        if i < len(x_msr_test) // batch_size:
            batch_x, batch_y = next(dincae_gen_10)
        else:
            # Special case for the last batch
            remaining_size = len(x_msr_test) % batch_size
            if remaining_size == 0:
                break
            batch_x, batch_y = next(dincae_gen_10)
            batch_x = batch_x[:remaining_size]
            batch_y = batch_y[:remaining_size]

        if is128:
            predictions = model_NW.predict(batch_x, verbose=0)[...,0]
        else:
            predictions = model.predict(batch_x, verbose=0)[...,0]

        # Denormalize data !!!
        predictions_denorm = predictions*residual_n_std + residual_n_mean
        true_values_denorm = batch_y[..., 0]*residual_n_std + residual_n_mean

        # Get the masks and calculate the errors
        diffMask = batch_y[..., 2]     #diff mask
        diff_errors_batch = np.where(diffMask, np.abs(predictions_denorm - true_values_denorm), np.nan)
        squared_errors = np.nanmean(diff_errors_batch**2, axis=(1, 2))# if np.any(~np.isnan(diff_errors_batch)) else np.nan
        #squared_errors = np.where(np.sum(diffMask,axis=(1,2))<100,np.nan,squared_errors)
        RMSE.append(np.sqrt(squared_errors))

    RMSE = np.concatenate(RMSE)
    all_RMSEs.append(np.nanmean(RMSE))
    all_batch_stds.append(np.nanstd(RMSE))

# Calculate the final RMSE and standard deviation
final_RMSE = np.nanmean(all_RMSEs)
final_std = np.nanstd(all_RMSEs)

# Print the results
print(f"Dincae 10% - Array of RMSEs: {all_RMSEs}")
print(f"Dincae 10% - Array of batch standard deviations: {all_batch_stds}")
print(f"Dincae 10% - Final result: RMSE = {final_RMSE} ± {final_std}")
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.interpolate import griddata
from scipy.interpolate import NearestNDInterpolator

# LOAD DATASETS
basepath = '/leonardo_scratch/fast/IscrC_AIWAF/angelo/'
# LOAD DATASETS

date_d = np.load(basepath+'datasets/date_d.npy')
date_n = np.load(basepath+'datasets/date_n.npy')

abs_dataset_d = np.load(basepath+'datasets/abs_dataset_d.npy')
abs_dataset_n = np.load(basepath+'datasets/abs_dataset_n.npy')

italy_mask = np.load(basepath+'datasets/italy_mask.npy')
# data_min = np.load(basepath+'datasets/data_min.npy')
# data_max = np.load(basepath+'datasets/data_max.npy')


# Cut the datasets to only the training data
if True:
    date_d = date_d[0:5832] # train from 2002-07-04 to 2018-07-04
    date_n = date_n[0:5832]
    abs_dataset_d = abs_dataset_d[0:5832]
    abs_dataset_n = abs_dataset_n[0:5832]
    # x_train_d = dataset_d[0:5832] # from 2002-07-04 to 2018-07-04
    # x_val_d = dataset_d[5832:6922] # from 2018-07-04 to 2021-07-04
    # x_test_d = dataset_d[6922:] # from 2021-07-04 to 2023-12-31


# Check data
def check_data(dataset):
    print('min:', np.nanmin(dataset))
    print('avg:', np.nanmean(dataset))
    print('max:', np.nanmax(dataset))

check_data(abs_dataset_d)
check_data(abs_dataset_n)

def nan_blur(a):
    sigma= 4/3                  # standard deviation for Gaussian kernel
    truncate=3.0               # truncate filter at this many sigmas

    #nana = np.sum(np.isnan(a))
    b=np.where(np.isnan(a),0,a)
    bb=ndimage.gaussian_filter(b,sigma=sigma,truncate=truncate) # blur the data
    #print(f"bb is {bb}")

    w=np.where(np.isnan(a),0.,1.)   # weight array
    #print(f"w is {w}")
    ww=ndimage.gaussian_filter(w,sigma=sigma,truncate=truncate)
    #print(ww)
    ww=np.where(ww==0,np.nan,ww)    # avoid divide by zero
    #print(ww)
    #nanww = np.sum(np.isnan(ww))
    #assert (nanww <= nana)
    res=bb/ww                    # weighted blur
    #nanres=np.sum(np.isnan(res))
    #assert (nanres == nanww)
    return(res)

if False:
    prova = np.ones((3,3))
    prova[:,1] = np.nan
    print(nan_blur(prova))
    assert False
        
def clean_dataset(dataset,ite=2):
    clean_dataset = np.copy(dataset)
    dim = dataset.shape[0]
    for i in range(dim):
        #print(i)
        mask = np.isnan(dataset[i])
        clouds = mask*italy_mask    # Exclude the landmass from the dilation
        dclouds = ndimage.binary_dilation(clouds,iterations=ite).astype(int)    # Dilate the clouds mask
        clean_dataset[i] = np.where((1-dclouds)*italy_mask, dataset[i], np.nan) # Apply the dilated mask
        #nmask =  np.isnan(cean_dataset[i])
        #blr = nan_blur(clean_dataset[i])
        #clean_dataset[i] = np.where(nmask,np.nan,blr)
    return(clean_dataset)
    
if True:
    clean_dataset_d = clean_dataset(abs_dataset_d)
    clean_dataset_n = clean_dataset(abs_dataset_n)
    check_data(clean_dataset_d)
    check_data(clean_dataset_n)
    


# ABSOLUTE BASELINE CREATION
# Create the ABS baseline arrays
abs_baseline_d = np.empty((366, 256, 256))
abs_baseline_n = np.empty((366, 256, 256))
abs_baseline_d_std = np.empty((366, 256, 256))
abs_baseline_n_std = np.empty((366, 256, 256))


# Convert the date arrays to pandas DatetimeIndex with datetime format, then get the day of the year for each date
date_series_d = pd.to_datetime(date_d, unit='D', origin=pd.Timestamp('1970-01-01'))
day_of_year_d = date_series_d.dayofyear
date_series_n = pd.to_datetime(date_n, unit='D', origin=pd.Timestamp('1970-01-01'))
day_of_year_n = date_series_n.dayofyear


# elementwise change where there are differences bigger than a certain amount (0.8, 2, 4... degrees)
# [x, y, z] -> combination of near bits for (x-1, x, x+1)
def remove_outlier_pixels(dataset, threshold=4):
    print("removing outliers")
    print(dataset.shape[0])
    for day in range(1, dataset.shape[0] - 1):  # First and last excluded

        min_day = np.clip(day - 1, 0, dataset.shape[0] - 1)
        max_day = np.clip(day + 1, 0, dataset.shape[0] - 1)

        current_day_data = dataset[day, :, :]
        prev_day_data = dataset[min_day, :, :]
        next_day_data = dataset[max_day, :, :]
        
        # Replace NaN values with the current day's data so they don't affect the calculations
        prev_day_data = np.where(np.isnan(prev_day_data), current_day_data, prev_day_data)
        next_day_data = np.where(np.isnan(next_day_data), current_day_data, next_day_data)
        
        # Calculate the absolute difference between the current day and the previous and next days
        diff_prev = np.abs(current_day_data - prev_day_data)
        diff_next = np.abs(current_day_data - next_day_data)

        if((diff_prev < threshold).all() and (diff_next < threshold).all()):
            continue
        
        # Count the outliers, i.e. the pixels where the difference is bigger than the threshold
        diff_prev_indices = np.where(diff_prev > threshold)
        diff_next_indices = np.where(diff_next > threshold)
        concat_array = list(zip(*diff_prev_indices)) + list(zip(*diff_next_indices))
        if(len(concat_array) > 0):
            print(f"found {len(concat_array)} outliers")


        # If the difference is bigger than the threshold, replace the current day's data with the average of the surrounding days
        for (x, y) in concat_array:
            today_value = dataset[day, x, y]

            if(np.isnan(today_value)):  # Skip NaNs
                print("nan found")
                continue

            # min_day = np.clip(day - 1, 0, dataset.shape[0] - 1)
            # max_day = np.clip(day + 1, 0, dataset.shape[0] - 1)
            # yesterday_value = np.where(np.isnan(dataset[min_day, x, y]), dataset[day, x, y], dataset[min_day, x, y])
            # tomorrow_value = np.where(np.isnan(dataset[max_day, x, y]), dataset[day, x, y], dataset[max_day, x, y])
            
            min_day = np.clip(day - 1, 0, dataset.shape[0] - 1)
            max_day = np.clip(day + 2, 0, dataset.shape[0] - 1)  # plus 2 to include the max
            min_x = np.clip(x - 1, 0, dataset.shape[1] - 1)
            max_x = np.clip(x + 2, 0, dataset.shape[1] - 1)
            min_y = np.clip(y - 1, 0, dataset.shape[2] - 1)
            max_y = np.clip(y + 2, 0, dataset.shape[2] - 1)
            
            adjacent_values = dataset[min_day:max_day, min_x:max_x, min_y:max_y]

            new_value = np.nanmean(adjacent_values) 
            dataset[day, x, y] = new_value  # Replace the outlier with the average of the surrounding days
    return dataset

if True:
    abs_dataset_d = remove_outlier_pixels(clean_dataset_d)
    abs_dataset_n = remove_outlier_pixels(clean_dataset_n)
    print("outliers removed")
    check_data(abs_dataset_d)
    check_data(abs_dataset_n)



# For each day of the year, calculate the average temperature for day and night
for day in range(0, 366):
    # Get the indices of the dates that match the current day of the year for day and night
    indices_d = np.where(day_of_year_d == day+1)    # Add 1 to day because day_of_year starts from 1
    indices_n = np.where(day_of_year_n == day+1)
    
    # Calculate the average temperature for the current day of the year for day and night, ignoring absent days
    mean_temp_d = np.nanmean(abs_dataset_d[indices_d], axis=0) if indices_d[0].size > 0 else np.nan
    mean_temp_n = np.nanmean(abs_dataset_n[indices_n], axis=0) if indices_n[0].size > 0 else np.nan
        
    # Assign the mean temperatures to the baseline arrays
    abs_baseline_d[day] = mean_temp_d
    abs_baseline_n[day] = mean_temp_n
    abs_baseline_d_std[day] = np.nanstd(abs_dataset_d[indices_d], axis=0) if indices_d[0].size > 0 else np.nan
    abs_baseline_n_std[day] = np.nanstd(abs_dataset_n[indices_n], axis=0) if indices_n[0].size > 0 else np.nan


abs_baseline_d_std = np.nanmean(abs_baseline_d_std)
abs_baseline_n_std = np.nanmean(abs_baseline_n_std)

print(f"mean std (global); d = {abs_baseline_d_std}, n = {abs_baseline_n_std}")

if False:
    abs_dataset_d_clean = np.copy(abs_dataset_d)
    abs_dataset_n_clean = np.copy(abs_dataset_n)

    for i in range(abs_dataset_d.shape[0]):
        day = day_of_year_d[i]-1
        abs_dataset_d_clean[i] = np.where(np.abs(abs_dataset_d[i] - abs_baseline_d[day])<4*abs_baseline_d_std, abs_dataset_d[i], np.nan)

    for i in range(abs_dataset_n.shape[0]):
        day = day_of_year_n[i]-1
        abs_dataset_n_clean[i] = np.where(np.abs(abs_dataset_n[i] - abs_baseline_n[day])<4*abs_baseline_n_std, abs_dataset_n[i], np.nan)

    removed_d = np.sum(np.isnan(abs_dataset_d_clean)) - np.sum(np.isnan(abs_dataset_d))
    removed_n = np.sum(np.isnan(abs_dataset_n_clean)) - np.sum(np.isnan(abs_dataset_n))

    print(f"removed {removed_d} data for day and {removed_n} data for night")
    print(f"percentage removed {removed_d/np.sum(~np.isnan(abs_dataset_d))*100} for day and {removed_n/np.sum(~np.isnan(abs_dataset_n))*100} for night")


# Create a meshgrid for the x and y coordinates, to be used in the interpolation
x = np.arange(256)
y = np.arange(256)
xx, yy = np.meshgrid(x, y)  # Grid of x, y coordinates

if True: #blur
    print(f"still remaining {np.sum(np.isnan(abs_baseline_d))} nans")
    abs_baseline_d = nan_blur(abs_baseline_d)
    abs_baseline_n = nan_blur(abs_baseline_n)
    print(f"still remaining {np.sum(np.isnan(abs_baseline_d))} nans")

    # Ensure that the land pixels are not modified
    for day in range(366):
        land_sum = np.sum(1-italy_mask)
        abs_baseline_d[day] = np.where(italy_mask,abs_baseline_d[day],np.nan)
        assert(np.sum(np.isnan(abs_baseline_d[day]))==land_sum)
    # Set the land pixels to value 0
    abs_baseline_d[:, ~italy_mask] = 0
    abs_baseline_n[:, ~italy_mask] = 0

    if True:
        # SAVE BASELINE ARRAYS
        np.save(basepath+'datasets/abs_custom4_baseline_d.npy', abs_baseline_d)
        np.save(basepath+'datasets/abs_custom4_baseline_n.npy', abs_baseline_n)

    assert False


# For all NaN values still present in the ocean, interpolate spatially
for day in range(366):
    for baseline_array in [abs_baseline_d, abs_baseline_n]:
        # Get the current baseline and create a mask for the valid values
        data = baseline_array[day]
        valid_mask = ~np.isnan(data) #& italy_mask

        # Get the valid values and their coordinates
        values = data[valid_mask]
        coords = np.array((xx[valid_mask], yy[valid_mask])).T   # Coordinates of the non-nan values, transposed back in 2D

        # Perform the interpolation only on the ocean pixels (italy_mask)
        data_interp = griddata(coords, values, (xx[italy_mask], yy[italy_mask]), method='linear')
        # Assign the interpolated data back to the ocean pixels in the baseline array
        baseline_array[day, italy_mask] = data_interp

        # Perform a nearest-neighbor interpolation to fill in any remaining NaN values
        valid_mask = ~np.isnan(baseline_array[day]) #& italy_mask
        values = baseline_array[day, valid_mask]
        coords = np.array((xx[valid_mask], yy[valid_mask])).T
        interpolator = NearestNDInterpolator(coords, values)    # Nearest-neighbor interpolator
        baseline_array[day, italy_mask] = interpolator((xx[italy_mask], yy[italy_mask]))    # Interpolation and assignment


# Put the value 0 in the land pixels of the baseline arrays
abs_baseline_d[:, ~italy_mask] = 0
abs_baseline_n[:, ~italy_mask] = 0

print(abs_baseline_d.shape)
print(abs_baseline_n.shape)


# # DEBUG : Iterate over the baseline arrays and check if there are still NaN values
# for i, baseline_array in enumerate([baseline_d, baseline_n, abs_baseline_d, abs_baseline_n]):
#     for day in range(baseline_array.shape[0]):
#         if np.isnan(baseline_array[day]).any():
#             print(f'NaN value found in baseline_array {i} on day {day}')
# #             plt.imshow(baseline_array[day])
# #             plt.colorbar()
# #             plt.title(f'Baseline_array {i} on day {day}')
# #             plt.show()

if True:
# SAVE BASELINE ARRAYS
    np.save('datasets/abs_new_baseline_d.npy', abs_baseline_d)
    np.save('datasets/abs_new_baseline_n.npy', abs_baseline_n)
    print("baseline saved")

assert False

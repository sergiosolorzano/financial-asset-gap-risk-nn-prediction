from sklearn.preprocessing import StandardScaler,MinMaxScaler
import sys
import os
import numpy as np
from math import floor

from sklearn.model_selection import KFold

#import scripts
import importlib as importlib
sys.path.append(os.path.abspath('./helper_functions_dir'))
import load_data as load_data
import generate_images as generate_images
import helper_functions as helper_functions

def generate_features_lists(stock_dataset_df, cols_used, transform_algo, transformed_img_sz, gaf_method, gaf_sample_range):
    #Generate images from dataset
    cols_used_count = sum(column_name in cols_used for column_name in stock_dataset_df.columns)
    #print("size df",len(stock_dataset_df))
    feature_image_dataset_list, feature_price_dataset_list, feature_label_dataset_list = generate_images.generate_multiple_feature_images(stock_dataset_df, cols_used, transform_algo, image_size=transformed_img_sz, method=gaf_method, gaf_sample_range=gaf_sample_range)
    #print("image data",feature_image_dataset_list,"labels",feature_label_dataset_list)
    print("shape [0] set",np.array(feature_image_dataset_list[0]).shape)

    #np.set_printoptions()

    return feature_image_dataset_list, feature_price_dataset_list, feature_label_dataset_list, cols_used_count

def squeeze_array(images_array, labels_array):
    #squeeze arrays
    images_array = images_array.squeeze(axis=(0, 1))
    print("len img",len(images_array),"image shape",images_array.shape)#,"prices_array[0][0]",prices_array[0][0])
    print("len label",len(labels_array),"labels shape",labels_array.shape)#,"prices_array[0][0]",prices_array[0][0])

    return images_array, labels_array

def Generate_feature_image_to_f32(labels_array, images_array, transformed_img_sz, scaler):
    feature_image_dataset_list_f32, labels_scaled_list_f32 = generate_images.Generate_feature_image_dataset_list_f32(labels_array, images_array, transformed_img_sz, scaler)
    return feature_image_dataset_list_f32, labels_scaled_list_f32

#create array of images
def create_images_array(feature_image_dataset_list, feature_label_dataset_list):
    images_array = helper_functions.data_to_array(feature_image_dataset_list)
    labels_array = helper_functions.data_to_array(feature_label_dataset_list)
    # images_array=np.array(feature_image_dataset_list)
    # labels_array=np.array(feature_label_dataset_list)
    #print("len price array",len(prices_array),prices_array.shape,prices_array)
    #print("images_array",len(images_array[0][0][0]),"labels_array",len(labels_array[0]),"prices array",len(prices_array[0][0]))
    #print("len img",len(images_array),"image shape",images_array.shape)#,"prices_array[0][0]",prices_array[0][0])
    #print("len label",len(labels_array),"labels shape",labels_array.shape)#,"prices_array[0][0]",prices_array[0][0])

    return images_array, labels_array

def conv_output_shape_dynamic(h_w, kernel_size=(1,1), stride=1):
        h = floor( (h_w[0] - kernel_size[0])/ stride) + 1
        w = floor( (h_w[1] - kernel_size[1])/ stride) + 1
        return h, w


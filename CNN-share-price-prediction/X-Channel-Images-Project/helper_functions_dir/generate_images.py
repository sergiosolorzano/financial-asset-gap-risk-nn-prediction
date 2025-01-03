from math import floor
from enum import Enum
import sys
import os
import time

import numpy as np

from sklearn.preprocessing import StandardScaler,MinMaxScaler

from pyts.image import GramianAngularField

from pyts.image import MarkovTransitionField

import importlib as importlib

import torchvision.transforms as transforms
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parameters import Parameters, TransformAlgo

def generate_transformed_images(dataset, transform_algo, gaf_img_sz=32, method="summation", gaf_sample_range=(0,1)):
    #print("len data series received:",len(dataset),"size",dataset.size)

    #determine num of gaf_img_szX images with gaf_img_sz datapoints
    num_images_to_generate = floor(len(dataset) / gaf_img_sz)
    #print("len dataset",len(dataset),"num_images_to_generate",num_images_to_generate)
    
    #reshape dataset into number of images
    dataset = dataset[:num_images_to_generate*gaf_img_sz].reshape(num_images_to_generate, gaf_img_sz)
    #print("data in GAF",dataset)
    
    transformed_images = None
    transformation_functions = {
        str(TransformAlgo.GRAMIAN): lambda dataset: GramianAngularField(image_size=gaf_img_sz, method=method, sample_range=gaf_sample_range).fit_transform(dataset),
        str(TransformAlgo.MARKOV): lambda dataset: MarkovTransitionField(image_size=gaf_img_sz).fit_transform(dataset)
    }
    try:
        #print("***transf algo", transform_algo, type(transform_algo), transform_algo.value)
        transformation_func = transformation_functions[str(transform_algo)]
    except KeyError:
        raise ValueError(f"Unsupported transformation algorithm: {transform_algo}")

    transformed_images = transformation_func(dataset)
    
    return transformed_images

def generate_multiple_feature_images(dataset, cols_used, transformed_algo, image_size=32, method="summation", gaf_sample_range = (0, 1)):
    
    feature_image_dataset_list=[[] for _ in range(len(cols_used))]
    feature_price_dataset_list=[[] for _ in range(len(cols_used))] #="Open", "High", "Low", "Close" , "Adj Close"
    feature_label_dataset_list=[] #next value for each chunk of ="Open", "High", "Low", "Close" , "Adj Close"
    column_idx = 0

    total_single_feature_chunks = 0

    for idx, column_name in enumerate(dataset.columns):

      #create open,  close, high, low images. 
      if column_name in cols_used:
        temp_image_list = []
        temp_price_list = []
        temp_label_list = []
        #print("dataset idx", idx, "len rows this data feature", len(dataset[column_name]), "dataset[i].shape", dataset[column_name].shape, "dataset i:", dataset[column_name])
        #print(f"Processing",column_name)

        full_feature_data = dataset[column_name].values
        full_feature_num_samples = len(full_feature_data)
        #print("dataset",full_feature_data)
        #print(f"full_feature_num_samples - col {column_name}",full_feature_num_samples)
        #if column_name == "Open": print("total input data",full_feature_data)

        #add 1 for last window label
        adj_feature_num_samples = full_feature_num_samples - (image_size + 1)
        num_windows = image_size
        #print("window size",adj_feature_num_samples)

        #loop by data_chunk so each chunk represents the price series that we slide by image_size
        #print("full data",full_feature_data)
        # TODO: parallelism
        #print(f"Total target windows:{num_windows}")
        for curr_window_index in range(num_windows):
          
          curr_sliding_window_data = full_feature_data[curr_window_index:adj_feature_num_samples+curr_window_index]
          #print(f"Curr window len {len(curr_sliding_window_data)} first value {curr_sliding_window_data[0]} next value {curr_sliding_window_data[1]} last value {curr_sliding_window_data[490]}")
          #if curr_window_index ==0 or curr_window_index ==1: print(f"Curr window len {len(curr_sliding_window_data)} input: {curr_sliding_window_data[:300]}")
          #if curr_window_index==1: print("window",curr_window_index,"curr_sliding_window_data",curr_sliding_window_data)
        
          target_num_chunks = floor(adj_feature_num_samples / image_size)
          #print(f"Target number of chunks for curr Window {column_name}",target_num_chunks)
          
          for cur_chunk in range(target_num_chunks):
            
            if column_name == "Open": total_single_feature_chunks += 1
            
            #chunk size of image size
            data_chunk = curr_sliding_window_data[cur_chunk*image_size:(cur_chunk*image_size)+image_size]
            #print("data chunk",cur_chunk*image_size,"to",(cur_chunk*image_size)+image_size,len(data_chunk))
            #if curr_window_index==1: print("window",curr_window_index,"data chunk",data_chunk)
            # if (cur_chunk < 5 and curr_window_index==0):
            #   print("cur_chunk",cur_chunk,"input chunk",data_chunk)
            #append gaf image to image list. store price feature values in price list
            transformed_images = generate_transformed_images(data_chunk, transformed_algo, gaf_img_sz=image_size, method=method, gaf_sample_range=gaf_sample_range)
            #print("gaf recevived",gaf_images)
            temp_image_list.append(transformed_images)
            #print("At chunk",cur_chunk,"input chunk size",len(data_chunk),"shape gaf images",gaf_images.shape, "len temp image list",len(temp_image_list))
            
            temp_price_list.append(curr_sliding_window_data[(cur_chunk*image_size)+image_size])
            # if (cur_chunk < 5 and curr_window_index==0):
            #   print("curr chunk",data_chunk)
            #   print("cur chunk label",curr_sliding_window_data[(cur_chunk*image_size)+image_size])
            # if(cur_chunk==0):
            #   print("Price Data Pre-Gaf: i", cur_chunk, "len",len(data_chunk), "shape", feature_data.shape, "data",data_chunk)
            #   print("Image Returned: idx", idx, "image size", gaf_images.size, f"first {image_size} image vals", gaf_images.flatten()[:image_size])
            
            #print("At chunk",cur_chunk,"input chunk size",len(data_chunk),"len price_list",len(price_list),price_list)
            
            #get next single value after the chunk as label to list
            #print("appending to temp label list-currcunk",cur_chunk,"imgsize",image_size,"labels",curr_sliding_window_data[(cur_chunk*image_size)+image_size])
            
            if Parameters.nn_predict_price:
                temp_label_list.append(curr_sliding_window_data[(cur_chunk*image_size)+image_size])
                #if curr_window_index==1: print("window",curr_window_index,"label",curr_sliding_window_data[(cur_chunk*image_size)+image_size])
            else:
                #next day price down
                if curr_sliding_window_data[(cur_chunk*image_size)+image_size] < curr_sliding_window_data[(cur_chunk*image_size)+image_size-1]:
                    temp_label_list.append(Parameters.classification_class_price_down)
                #next day price up
                else:
                    temp_label_list.append(Parameters.classification_class_price_up)
            #feature_label_index_dataset_list.append(feature_data[cur_chunk + image_size + 1])
            #print("chunk",cur_chunk,"label for",column_name,"price",feature_data[cur_chunk + image_size + 1])
            #if(column_name == "Open"):
              #index position for the label of this chunk
              #feature_label_index_dataset_list.append(cur_chunk + image_size + 1)
              #print("at chunk",cur_chunk,"feature label list",feature_label_index_dataset_list)
        #time.sleep(10) 
        
        #if column_name == "Open": print("total chunks Open feature:", total_single_feature_chunks)
        #print(f"Column {column_name} temp image list len to append",len(temp_image_list))
        feature_image_dataset_list[column_idx].append(temp_image_list)
        #print("feature_image_dataset_list",feature_image_dataset_list)
        feature_price_dataset_list[column_idx].append(temp_price_list)
        #print("price list",price_list)
        feature_label_dataset_list.append(temp_label_list)
        column_idx += 1

    print("Final len images",len(feature_image_dataset_list),
            "len image list index (i.e. feature) 0",len(feature_image_dataset_list[0][0]))
    # print("Final len price list",len(feature_price_dataset_list),
    #        "len feature_price_dataset_list index 0 (i.e. column 0)", len(feature_price_dataset_list[0][0]),feature_price_dataset_list)
    # print("Final len labels", len(feature_label_dataset_list),feature_label_dataset_list) # 2455=total range*5
    
    feature_image_dataset_list = np.array(feature_image_dataset_list) 
    #print("Final Shape of images before transpose:", feature_image_dataset_list.shape, feature_image_dataset_list)
    
    #transpose image for CNN
    #(5, 1, 491, 1, 32, 32)
    print("Shape before transpose:", feature_image_dataset_list.shape)
    feature_image_dataset_list= np.transpose(feature_image_dataset_list, (1, 3, 0, 2, 4, 5))
    #print("Final Shape of images after transpose:", feature_image_dataset_list.shape)
    
    return feature_image_dataset_list, feature_price_dataset_list, feature_label_dataset_list

def generate_multiple_feature_images_myoverlap(dataset, cols_used, transformed_algo, image_size=32, overlap = 20, method="summation", gaf_sample_range = (0, 1)):
    
    feature_image_dataset_list=[[] for _ in range(len(cols_used))]
    feature_price_dataset_list=[[] for _ in range(len(cols_used))] #="Open", "High", "Low", "Close" , "Adj Close"
    feature_label_dataset_list=[] #next value for each chunk of ="Open", "High", "Low", "Close" , "Adj Close"
    column_idx = 0

    total_single_feature_chunks = 0

    for idx, column_name in enumerate(dataset.columns):

      #create open,  close, high, low images. 
      if column_name in cols_used:
        temp_image_list = []
        temp_price_list = []
        temp_label_list = []
        #print("dataset idx", idx, "len rows this data feature", len(dataset[column_name]), "dataset[i].shape", dataset[column_name].shape, "dataset i:", dataset[column_name])
        #print(f"Processing",column_name)

        full_feature_data = dataset[column_name].values
        full_feature_num_samples = len(full_feature_data)
        #print("dataset",full_feature_data)
        #print(f"full_feature_num_samples - col {column_name}",full_feature_num_samples)
        #if column_name == "Open": print("total input data",full_feature_data)

        #add 1 for last window label
        adj_feature_num_samples = full_feature_num_samples - (image_size + 1)
        #print("data",full_feature_data)
        num_windows = image_size
        #print("window size",adj_feature_num_samples)

        #loop by data_chunk so each chunk represents the price series that we slide by image_size
        #print("full data",full_feature_data)
        # TODO: parallelism
        #print(f"Total target windows:{num_windows}")
        for curr_window_index in range(num_windows):
          
          curr_sliding_window_data = full_feature_data[curr_window_index:adj_feature_num_samples+curr_window_index]
          #print(f"Curr window len {len(curr_sliding_window_data)} first value {curr_sliding_window_data[0]} next value {curr_sliding_window_data[1]} last value {curr_sliding_window_data[490]}")
          #if curr_window_index ==0 or curr_window_index ==1: print(f"Curr window len {len(curr_sliding_window_data)} input: {curr_sliding_window_data[:300]}")
          #if curr_window_index==1: print("window",curr_window_index,"curr_sliding_window_data",curr_sliding_window_data)
        
          target_num_chunks = floor(adj_feature_num_samples / (image_size+overlap))
          #print(f"Target number of chunks for curr Window {column_name}",target_num_chunks)
          
          step_size = image_size - overlap
          counter = 0
          for start_pos in range(0, adj_feature_num_samples, step_size):
            
            if start_pos + image_size < len(full_feature_data):
                
                if column_name == "Open": total_single_feature_chunks += 1
                
                #chunk size of image size
                data_chunk = full_feature_data[start_pos:start_pos + image_size]
                #print("curr_window_index",curr_window_index,"chunk",data_chunk)
                #data_chunk = curr_sliding_window_data[cur_chunk*(image_size):(cur_chunk*image_size)+image_size]
                #print("data chunk",cur_chunk*image_size,"to",(cur_chunk*image_size)+image_size,len(data_chunk))
                #if curr_window_index==1: print("window",curr_window_index,"data chunk",data_chunk)
                # if (cur_chunk < 5 and curr_window_index==0):
                #   print("cur_chunk",cur_chunk,"input chunk",data_chunk)
                #append gaf image to image list. store price feature values in price list
                transformed_images = generate_transformed_images(data_chunk, transformed_algo, gaf_img_sz=image_size, method=method, gaf_sample_range=gaf_sample_range)
                #print("gaf recevived",gaf_images)
                temp_image_list.append(transformed_images)
                #print("At chunk",cur_chunk,"input chunk size",len(data_chunk),"shape gaf images",gaf_images.shape, "len temp image list",len(temp_image_list))
                
                #temp_price_list.append(curr_sliding_window_data[(cur_chunk*image_size)+image_size])
                temp_price_list.append(full_feature_data[start_pos + image_size])
                # if (cur_chunk < 5 and curr_window_index==0):
                #   print("curr chunk",data_chunk)
                #   print("cur chunk label",curr_sliding_window_data[(cur_chunk*image_size)+image_size])
                # if(cur_chunk==0):
                #   print("Price Data Pre-Gaf: i", cur_chunk, "len",len(data_chunk), "shape", feature_data.shape, "data",data_chunk)
                #   print("Image Returned: idx", idx, "image size", gaf_images.size, f"first {image_size} image vals", gaf_images.flatten()[:image_size])
                
                #print("At chunk",cur_chunk,"input chunk size",len(data_chunk),"len price_list",len(price_list),price_list)
                
                #get next single value after the chunk as label to list
                #print("appending to temp label list-currcunk",cur_chunk,"imgsize",image_size,"labels",curr_sliding_window_data[(cur_chunk*image_size)+image_size])
                
                if Parameters.nn_predict_price:
                    label_index = start_pos + image_size
                    #if label_index < len(full_feature_data):  # Ensure we don't go out of bounds
                    temp_label_list.append(full_feature_data[label_index])
                    #print("window",curr_window_index,"counter",counter,"label",full_feature_data[label_index])
                    #temp_label_list.append(curr_sliding_window_data[(cur_chunk*image_size)+image_size])
                    #if curr_window_index==1: print("window",curr_window_index,"label",curr_sliding_window_data[(cur_chunk*image_size)+image_size])
                else:
                    #next day price down
                    if curr_sliding_window_data[(start_pos*image_size)+image_size] < curr_sliding_window_data[(start_pos*image_size)+image_size-1]:
                        temp_label_list.append(Parameters.classification_class_price_down)
                    #next day price up
                    else:
                        temp_label_list.append(Parameters.classification_class_price_up)
                #feature_label_index_dataset_list.append(feature_data[cur_chunk + image_size + 1])
                #print("chunk",cur_chunk,"label for",column_name,"price",feature_data[cur_chunk + image_size + 1])
                #if(column_name == "Open"):
                #index position for the label of this chunk
                #feature_label_index_dataset_list.append(cur_chunk + image_size + 1)
                #print("at chunk",cur_chunk,"feature label list",feature_label_index_dataset_list)

                counter+=1
        #time.sleep(10) 
        
        #if column_name == "Open": print("total chunks Open feature:", total_single_feature_chunks)
        #print(f"Column {column_name} temp image list len to append",len(temp_image_list))
        feature_image_dataset_list[column_idx].append(temp_image_list)
        #print("feature_image_dataset_list",feature_image_dataset_list)
        feature_price_dataset_list[column_idx].append(temp_price_list)
        #print("price list",price_list)
        feature_label_dataset_list.append(temp_label_list)
        column_idx += 1

    print("Final len images",len(feature_image_dataset_list),
            "len image list index (i.e. feature) 0",len(feature_image_dataset_list[0][0]))
    # print("Final len price list",len(feature_price_dataset_list),
    #        "len feature_price_dataset_list index 0 (i.e. column 0)", len(feature_price_dataset_list[0][0]),feature_price_dataset_list)
    # print("Final len labels", len(feature_label_dataset_list),feature_label_dataset_list) # 2455=total range*5
    
    feature_image_dataset_list = np.array(feature_image_dataset_list) 
    #print("Final Shape of images before transpose:", feature_image_dataset_list.shape, feature_image_dataset_list)
    
    #transpose image for CNN
    #(5, 1, 491, 1, 32, 32)
    print("Shape before transpose:", feature_image_dataset_list.shape)
    feature_image_dataset_list= np.transpose(feature_image_dataset_list, (1, 3, 0, 2, 4, 5))
    #print("Final Shape of images after transpose:", feature_image_dataset_list.shape)
    
    return feature_image_dataset_list, feature_price_dataset_list, feature_label_dataset_list

def generate_multiple_feature_images_overlap(dataset, cols_used, transformed_algo, image_size=32, overlap = 20, method="summation", gaf_sample_range = (0, 1)):
    
    feature_image_dataset_list = [[] for _ in range(len(cols_used))]
    feature_price_dataset_list = [[] for _ in range(len(cols_used))] 
    feature_label_dataset_list = [] 
    column_idx = 0
    total_single_feature_chunks = 0

    step_size = image_size - overlap  # Adjust the step size to account for the 20-day overlap

    for idx, column_name in enumerate(dataset.columns):
        if column_name in cols_used:
            #print("***Col Used",column_name)
            temp_image_list = []
            temp_price_list = []  # This will store the prices for each window
            temp_label_list = []
            full_feature_data = dataset[column_name].values
            full_feature_num_samples = len(full_feature_data)

            #print("dataset",full_feature_data[0:100])

            #ensure there's enough data for the last window
            adj_feature_num_samples = full_feature_num_samples - image_size
            
            # Loop by data_chunk so each chunk represents the price series that we slide by step_size
            for curr_window_index in range(0, adj_feature_num_samples, step_size):
                curr_sliding_window_data = full_feature_data[curr_window_index:curr_window_index + image_size]

                #print("cur window data",curr_sliding_window_data)
                #check enough data
                if len(curr_sliding_window_data) == image_size:
                    
                    transformed_images = generate_transformed_images(curr_sliding_window_data, transformed_algo, gaf_img_sz=image_size, method=method, gaf_sample_range=gaf_sample_range)
                    temp_image_list.append(transformed_images)

                    temp_price_list.append(curr_sliding_window_data[-1])

                    if Parameters.nn_predict_price:
                        #print(f"label {total_single_feature_chunks}",full_feature_data[curr_window_index + image_size])
                        temp_label_list.append(full_feature_data[curr_window_index + image_size])
                    else:
                        if full_feature_data[curr_window_index + image_size] < full_feature_data[curr_window_index + image_size - 1]:
                            temp_label_list.append(Parameters.classification_class_price_down)
                        else:
                            temp_label_list.append(Parameters.classification_class_price_up)

                    if column_name == "Open":
                        total_single_feature_chunks += 1

            #print("total_single_feature_chunks open",total_single_feature_chunks)
            feature_image_dataset_list[column_idx].append(temp_image_list)
            feature_price_dataset_list[column_idx].append(temp_price_list)  # Append the price list for this column
            feature_label_dataset_list.append(temp_label_list)
            column_idx+=1

    print("num lists train",len(feature_image_dataset_list[0][0]))
    feature_image_dataset_list = np.array(feature_image_dataset_list)

    # Transpose image for CNN
    print("Shape before transpose:", feature_image_dataset_list.shape)
    feature_image_dataset_list = np.transpose(feature_image_dataset_list, (1, 3, 0, 2, 4, 5))
    
    return feature_image_dataset_list, feature_price_dataset_list, feature_label_dataset_list

def Generate_feature_image_dataset_list_f32(labels_array, images_array, image_size, scaler):
    # print("Scaler received",scaler)
    # print("labels_array",labels_array[:1])
    feature_image_dataset_list_f32 = np.array(images_array).astype(np.float32)
    feature_image_dataset_list_f32 = feature_image_dataset_list_f32.reshape(-1, image_size, image_size)
    #images_array = np.transpose(feature_image_dataset_list, (1, 0, 2, 3))

    labels_array = np.array(labels_array)
    #print("labels array",labels_array)
    reshaped_labels_array = labels_array.reshape(-1, 1)
    #print("reshaped labels array",reshaped_labels_array)
    
    #scale labels if regression prediction
    if Parameters.nn_predict_price==1:
        labels_scaled_list_f32 = scaler.fit_transform(reshaped_labels_array).reshape(-1,).astype(np.float32)
    else:
        labels_scaled_list_f32 = reshaped_labels_array.reshape(-1,).astype(np.float32)
    #print("scaled labels",labels_scaled_list_f32)
    print("4D image array shape",images_array.shape)
    print("3D reshaped image array ",feature_image_dataset_list_f32.shape)
    print("labels shape",reshaped_labels_array.shape)
    #print("Print Labels",labels_scaled_list_f32[:1])
    
    return feature_image_dataset_list_f32, labels_scaled_list_f32

def SetTransform(normalize_ftor=0.5,resolution_x=32,resolution_y=32):
    return transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([normalize_ftor], [normalize_ftor])
    #transforms.Resize((resolution_x, resolution_y))
    ])
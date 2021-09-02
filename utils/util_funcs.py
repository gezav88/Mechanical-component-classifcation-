import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#from matplotlib import pyplot as plt
import pymesh
import pickle
import pandas as pd 
import json


def plot_distribution(data_dir, top_x):
    train_ = []
    test_ = []
    map_classes = {}
    final_dict = {}
    
    train_files_list = glob.glob(os.path.join(data_dir, "train/*/"))
    
    # extract points from testing set:
    test_files_list = glob.glob(os.path.join(data_dir, "test/*/"))
    
    for i, folder_class in enumerate(train_files_list):
        map_classes[i] = folder_class.split("/")[-1]
        objects_3d_inside_class = [f for f in os.listdir(folder_class) if os.path.isfile(os.path.join(folder_class, f))]
        dist = len(objects_3d_inside_class)
        final_dict[map_classes[i][6:-1]] = dist
        #print(final_dict)
        df = pd.DataFrame(final_dict.items(), columns = ["class", "quantity_images"])
        
    plt.figure(figsize=(12,14))
    # select top 15 classes: 
    plt.barh(df["class"], df["quantity_images"], align='center', alpha=0.5)
    #plt.xticks(distribution["class"], objects)
    plt.ylabel('quantity Images')
    plt.title('dataset classes')
    plt.tick_params(axis='both', which='major', labelsize=12)
        
    top_15 = df.sort_values("quantity_images", ascending=False).head(int(top_x)).reset_index(drop=True)
    top_15_list = top_15["class"].tolist()
        
    return df, top_15_list

def unique(list1):
    x = np.array(list1)
    print(np.unique(x))
    
def save_array(data, name_to_save):
    np.save(name_to_save+".npy", data)
    
def load_arrays(data_name):
    data = np.load(data_name+".npy")
    
    return data

def load_dict(dict_name):
    # reading the data from the file
    with open(dict_name+'.txt') as f:
        data = f.read()
    # reconstructing the data as a dictionary
    dict_out = json.loads(data)
    
    return dict_out
    
def save_dict(data, name_to_save):
    with open(name_to_save, 'w') as f:
        f.write(json.dumps(data))
        
def save_list(data, listname):
    with open(listname+'.txt', 'w') as file:
        for listitem in data:
            file.write('%s\n' % listitem)
            
def load_list(filelist):
    outList = []
    # open file and read the content in a list
    with open(filelist+'.txt', 'r') as filehandle:
        outList = [item.rstrip() for item in filehandle.readlines()]
    
    return outList
        
def extract_points_from_obj(dir_to_extract, sample_points, top_list, train):
    points_list = []
    points_labels = []
    map_full_classes_train = {}
    map_full_classes_test = {}
    
    reduc_class = 0
    for i, folder_class in enumerate(dir_to_extract):
        if train == True:
            add_class_train = folder_class.split("/")[-1]
            map_full_classes_train[i] = add_class_train[6:-1]
            objects_3d_inside_class = [f for f in os.listdir(folder_class) if os.path.isfile(os.path.join(folder_class
                                                                                                              , f))]
            if map_full_classes_train[i] in top_list: # map_classes[i][6:-1]
                print("processing Train class:  {}".format(map_full_classes_train[i]))
                for objects in objects_3d_inside_class:
                    #print(objects)
                    mesh_i = trimesh.load(folder_class+objects)
                    try:
                        points_i = mesh_i.sample(sample_points)
                        points_list.append(points_i)
                        points_labels.append(i)
                    except:
                        pass
            else:
                pass
            objects_3d_inside_class = []
            # increase counter to map_reduced classes:
        else:
            add_class_test = folder_class.split("/")[-1]
            map_full_classes_test[i] = add_class_test[5:-1]
            objects_3d_inside_class = [f for f in os.listdir(folder_class) if os.path.isfile(os.path.join(folder_class
                                                                                                          , f))]
            if map_full_classes_test[i]  in top_list: #map_classes[i][5:-1]
                print("processing Test class:  {}".format(map_full_classes_test[i]))
                
                for objects in objects_3d_inside_class:
                    #print(objects)
                    mesh_i = trimesh.load(folder_class+objects)
                    try:
                        points_i = mesh_i.sample(sample_points)
                        points_list.append(points_i)
                        points_labels.append(i)
                    except:
                        pass
            else:
                pass
            objects_3d_inside_class = []
            
    
    return points_list, points_labels, map_full_classes_train

def mesh_to_points(data_dir,  top_classes, num_points_to_sample=2048):
    
    # Define the lists to save point cloud and label on the training and test set:
    train_cloud = []
    train_label = []
    test_cloud = []
    test_label = []
    
    class_dir = {}
    
    # Check if files exist in local computer:
    if (os.path.isfile('train_points.npy')) & (os.path.isfile('train_points.npy')):
        print("Loading files from local")
        top_classes = None
        # train:
        train_cloud_ar = load_arrays('train_points')
        new_train_label_array = load_arrays('train_labels')
        
        dict_map_reduced = load_dict(dict_name = 'class_map_reduced')
        
        # test:
        test_cloud_ar = load_arrays('test_points')
        new_test_label_array = load_arrays('test_labels')
        
        num_classes = len(dict_map_reduced)
        
    else:
        print ("Running process: ")
    
        print("Directory to look for training and testing sets: ", data_dir)
        # extract points from training set:
        train_files_list = glob.glob(os.path.join(data_dir, "train/*/"))
        # Extract classes from training folder:
        num_classes = len(train_files_list)
        print("\nNumber of classes in set: {}".format(num_classes))
        print("\n")
        #######################################################################################################
        # call function to extract points:
        
        train_cloud, train_label, class_dir_train  = extract_points_from_obj(
                                                                               dir_to_extract = train_files_list
                                                                             , sample_points=num_points_to_sample
                                                                             , top_list = top_classes
                                                                             , train = True)

        train_cloud_ar = np.array(train_cloud)
        save_array(data = train_cloud_ar, name_to_save = 'train_points')


        

        save_dict(data = class_dir_train, name_to_save = 'class_map_full.txt')
        
        
        dict_map_reduced = {}
        list_uniques = list(set(train_label))
        for i in np.arange(0, len(list_uniques), 1):
            index = list_uniques.index(list_uniques[i])
            dict_map_reduced[list_uniques[i]] = index

        new_array_out = [dict_map_reduced.get(n, n) for n in train_label]
        #print(new_array_out)
        new_train_label_array = np.array(new_array_out)
        save_array(data = new_train_label_array, name_to_save = 'train_labels')
        
        save_dict(data = dict_map_reduced, name_to_save = 'class_map_reduced.txt')
        
        #######################################################################################################
        # extract points from testing set:
        test_files_list = glob.glob(os.path.join(data_dir, "test/*/"))
        test_cloud, test_label, class_dir_test = extract_points_from_obj(dir_to_extract = test_files_list
                                                                         , sample_points=num_points_to_sample
                                                                         , top_list = top_classes
                                                                         , train = False) 
        test_cloud_ar = np.array(test_cloud)
        save_array(data = test_cloud_ar, name_to_save = 'test_points')
        
        dict_map_reduced_test = {}
        list_uniques_test = list(set(test_label))
        for i in np.arange(0, len(list_uniques_test), 1):
            index = list_uniques_test.index(list_uniques_test[i])
            dict_map_reduced_test[list_uniques_test[i]] = index

        new_array_out_test = [dict_map_reduced_test.get(n, n) for n in test_label]
        #print(new_array_out)
        new_test_label_array = np.array(new_array_out_test)
        save_array(data = new_test_label_array, name_to_save = 'test_labels')
        #########################################################################################################
    
    return (train_cloud_ar , new_train_label_array , test_cloud_ar , new_test_label_array , dict_map_reduced , num_classes)


def process_tf_data(points, label):
    # jitter points separate points so they are not on top of each other:
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    
    # shuffle points
    points = tf.random.shuffle(points)
    
    return points, label
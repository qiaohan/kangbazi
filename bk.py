import configuration 

import keras
from keras.models import Model
import tensorflow as tf
import keras.applications.xception as xception

import numpy as np

import os
from tqdm import tqdm
from time import time

import cv2
#import prepare_data

## image related functions
def resize_img_reflection_padding(img, new_size = 299):
    ''' 
    pad border with REFLECTION to maintain the overall style(color, lighterning...) of the image
    new_size: input size of the pre-trained model. 
    image   : individual (xx, xx, 3) jpg
    return the image.
    ''' 
    imsize = img.shape[0:2]
    border_width = np.abs(imsize[0] - imsize[1])

    if imsize[0] > imsize[1]:
        img_ = cv2.copyMakeBorder(img, 0,0,border_width//2, (border_width + 1)//2,cv2.BORDER_REFLECT)
    else:
        img_ = cv2.copyMakeBorder(img, border_width//2, (border_width + 1)//2, 0,0, cv2.BORDER_REFLECT)
    img = cv2.resize(img_,(new_size, new_size))
    return img

## log related functions
def feature_extraction_utils_loadlog(log_folder):
    if not os.path.exists(log_folder):
        return []
    
    fn = os.listdir(log_folder)
    if not fn: return []
        
    log_file = os.path.join(log_folder, fn[0])
    with open(log_file, 'r') as file:
        image_id_extracted_with_EOD = list(file.readlines())
        image_id_extracted = [item[:-1] for item in image_id_extracted_with_EOD]
    return image_id_extracted

    
def feature_extraction_utils_writelog(image_id_ls, log_folder, recover_log_flag = False, feature_folder = []):
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
        print('folder is created to store the log of feature-extracted image: {}'.format(log_folder))
    log_file = os.path.join(log_folder, 'processed_image_id.txt')

    if recover_log_flag:
        image_id_ls_with_EOS =  [filename.split('.')[0] + '\n' for filename in os.listdir(feature_folder)]
        with open(log_file, 'w') as file:
            file.writelines(image_id_ls_with_EOS)
    else:
        with open(log_file, 'a') as file:
            image_id_ls_with_EOS = [item  + '\n' for item in image_id_ls]
            file.writelines(image_id_ls_with_EOS)
    return 

def feature_extraction_utils_del_image_ifprocessed(image_id_ls_ori, log_folder = configuration.image_folder):
    image_id_preprocessed =  feature_extraction_utils_loadlog(log_folder)
    ## mlog(n) to finish this. time-cosuming. no better way so far.
    for img_id in image_id_preprocessed:
        image_id_ls_ori.remove(img_id)

## model related functions
def model():
    ## set up the mode
    base_model  = xception.Xception(include_top = False, weights = 'imagenet')
    conv_output = base_model.get_layer('block14_sepconv2_act').output
    # output size (None, 2, 2, 2048)
    feature     = keras.layers.MaxPooling2D(pool_size = (5, 5), strides = 5, padding = 'valid')(conv_output)
    feature_extractor = Model(inputs = base_model.input, outputs = feature)
    
    return feature_extractor

def prepare_image_batch(image_id_ls, image_folder = configuration.image_folder):
    image_ls = []
    for image_id in image_id_ls:
        fn   = os.path.join(image_folder, image_id + '.jpg')
        #print("preprocessing ",image_id)
        try:
            img_ = cv2.imread(fn)
            img  =  resize_img_reflection_padding(img_,  configuration.img_size)
            # will preprocess layer in pretrained model take care of this? check with tensorboard, and look at the graph.
        except:
            print(image_id," is bad!")
            img = np.zeros([configuration.img_size, configuration.img_size, 3])
        image_ls.append(np.array(img, np.float32))
        
    image_np = np.array(image_ls)
    return image_np

def extract_feature_batch(image_id_ls, image_folder, feature_extractor):
    image_np        = prepare_image_batch(image_id_ls, image_folder)
    image_processed = xception.preprocess_input(image_np)
    feature         = feature_extractor.predict(image_processed)
    return feature


def save_feature_batch(image_id_ls, feature, feature_folder = configuration.feature_folder):
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)
        print('folder is created to store the extracted image feature : {}'.format(feature_folder))
        
    ii = 0
    for image_id in image_id_ls:
        fn = os.path.join(feature_folder, image_id)
        np.save(fn, feature[ii])
        ii += 1
        
def main():
    # load parameter
    start_from_checkpoint = True

    batch_size     = configuration.batch_size   
    image_folder   = configuration.image_folder
    log_folder     = configuration.log_folder
    feature_folder = configuration.feature_folder
    
    # prepare the model
    feature_extractor = model() 

    image_id_ls_all = [filename.split('.')[0] for filename in os.listdir(image_folder)]
    if start_from_checkpoint:
        feature_extraction_utils_del_image_ifprocessed(image_id_ls_all, log_folder)


    total_time = 0

    for ii in tqdm(range(0, len(image_id_ls_all), batch_size)):
        image_id_ls = image_id_ls_all[ii: min(ii + batch_size, len(image_id_ls_all))]

        start = time()
        
        feature = extract_feature_batch(image_id_ls, image_folder, feature_extractor)
        save_feature_batch(image_id_ls, feature, feature_folder)
        feature_extraction_utils_writelog(image_id_ls, log_folder)

        total_time +=  time() - start
    print('It takes {} second in total, {} second per 1000 image on average'.format(total_time, total_time * 1000/ len(image_id_ls_all)))

if __name__ == "__main__":
    main()

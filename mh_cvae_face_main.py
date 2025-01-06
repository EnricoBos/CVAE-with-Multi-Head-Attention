# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 10:00:49 2025

@author: Enrico
"""

from pathlib import Path
import pandas as pd
import os 
import numpy as np
import imageio               # For reading image files
from PIL import Image         # For resizing images
import logging
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import json
import t_cvae_class
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

###############################################################################
# GPU distribution setup ######################################################
def setup_strategy():
    gpus = tf.config.list_physical_devices('GPU')  # Avoid using the deprecated experimental API

    if gpus:
        logger.info(f"Detected {len(gpus)} GPU(s). Setting up distribution strategy.")
        try:
            # Set memory growth and visible devices before anything else
            tf.config.set_visible_devices(gpus[0], 'GPU')
            #tf.config.experimental.set_memory_growth(gpus[0], True)
            
            # Set up the strategy (single GPU in this case)
            strategy = tf.distribute.get_strategy()  # Single-device strategy
        except RuntimeError as e:
            logger.error(f"Error configuring GPU: {e}")
            strategy = tf.distribute.get_strategy()  # Fallback to default strategy
    else:
        logger.info("No GPUs detected. Using default strategy.")
        strategy = tf.distribute.get_strategy()  # Single-device strategy
    
    return strategy


###############################################################################
def fetch_dataset(dx=80, dy=80, dimx=48, dimy=48, DATASET_PATH='', TXT_FILE_ATTR='')-> np.ndarray:
    """
    Fetches and processes images from the specified dataset directory.
    
    Parameters:
    dx (int): Number of pixels to crop from the left and right sides of the image.
    dy (int): Number of pixels to crop from the top and bottom of the image.
    dimx (int): The width to resize the image to after cropping.
    dimy (int): The height to resize the image to after cropping.
    DATASET_PATH (str): Path to the directory containing the dataset of images.
    
    Returns:
    np.ndarray: An array of processed images after resizing and cropping.
    """
    
    
    df_attrs = pd.read_csv(TXT_FILE_ATTR, sep='\t', skiprows=1,) 
    df_attrs = pd.DataFrame(df_attrs.iloc[:,:-1].values, columns = df_attrs.columns[1:])
    photo_ids_ = []; all_photos = []; all_attrs=[]
    seen_paths = set()  # Set to track already seen file paths
    for dirpath, dirnames, filenames in os.walk(str(DATASET_PATH)):
        for fname in filenames:
            if fname.endswith(".jpg"):
                fpath = os.path.join(dirpath,fname)
                # Check if the path has already been seen
                if fpath in seen_paths:
                    continue  # Skip if already processed
                # Add the path to the seen set
                seen_paths.add(fpath)
                photo_id = fname[:-4].replace('_',' ').split()
                person_id = ' '.join(photo_id[:-1])
                photo_number = int(photo_id[-1])
                photo_ids_.append({
                            'person':person_id,
                            'imagenum':photo_number,
                            'photo_path':fpath})
               
    df_photo_ids = pd.DataFrame(photo_ids_)
    df = pd.merge(df_photo_ids,df_attrs,on=('person','imagenum'), how='inner')
    ### some data are missing --> duplicated the images when needed !
    assert len(df)==len(df_attrs),"lost some data when merging dataframes"
   
    all_photos = df['photo_path'].apply(imageio.imread)\
                                .apply(lambda img:img[dy:-dy,dx:-dx])\
                                .apply(lambda img: np.array(Image.fromarray(img).resize([dimx,dimy])) )
   
    all_photos = np.stack(all_photos.values).astype('uint8')

    # ##DBG: Visualize one image (e.g., the first one) --> uncomment if needed !
    # Folder to save images
    output_folder = 'images_saved'
    os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist
    
    # Select 4 random images
    num_images = len(all_photos)
    random_indices = np.random.choice(num_images, size=4, replace=False)  # Select 4 unique random indices
    
    # Save selected images
    for i, idx in enumerate(random_indices, start=1):
        image = all_photos[idx]  # Select image at random index
        file_name = os.path.join(output_folder, f'processed_original_{i}.png')  # Create file path
    
        # Save image as PNG
        img = Image.fromarray(image)
        img.save(file_name)
    
        print(f"Saved: {file_name}")
    
    
    return all_photos, df


###############################################################################
def runner(experiments,
           data_img,
           df_attrs ,
           data_val,
           data_attrs_val ,
           enable_trainig,
           path_save_checkpoint
           ):
    
    logger.info('Starting train with experiments..')
    for exp_name, params in experiments.items():
        logger.info(f"Processing c_vae experiment '{exp_name}' with parameters: {params}")
        model_weights = None  # Initialize cvae to None for each experiment
        
        logger.info(f"Starting cvae experiment: {exp_name}")
        input_dim = data_img.shape[1]
        IMAGE_H = data_img.shape[1]
        IMAGE_W = data_img.shape[2]
        N_CHANNELS = 3
        
        label_dim = df_attrs.shape[1]
        if(enable_trainig): # start run
            logger.info("Starting training.....")
            model_weights = run_cvae(
                data_img,
                df_attrs,
                data_val,
                data_attrs_val ,
                IMAGE_H,
                IMAGE_W ,
                label_dim ,
                checkpoint_dir=path_save_checkpoint /f"{exp_name}_cvae_best_weights/", 
                **params ) # Unpack parameters
        else:
           
            logger.info("Loading CVAE model...")
 
            
            pattern = str(path_save_checkpoint /f"{exp_name}_cvae_best_weights/")
            # Search for the directory matching the pattern in the given directory
            weights_file = os.path.isdir(pattern)
            
            if weights_file:
                # Extract configuration for the model
                latent_dim = params['latent_dim']
                learning_rate = params['learning_rate']
                encoder_filters = tuple(params['encoder_filters'])
                decoder_filters = tuple(params['decoder_filters'])
                
                cvae = t_cvae_class.ConditionalVariationalAutoencoder(
                                                                        IMAGE_H,
                                                                        IMAGE_W ,
                                                                        label_dim,
                                                                        latent_dim, 
                                                                        learning_rate,
                                                                        encoder_filters,
                                                                        decoder_filters)
                # Load checkpoint before training
                cvae.load_checkpoint(pattern)
                logger.info(f"Successfully loaded weights for CVAE {exp_name}")
                
                #### visualize an image using randon value
                
                batch = 4
                z = np.random.randn(batch, latent_dim)
                # Generate random array
                #### example (1,1) = sunglasses and smiling , (0,0) = no smiling and no sunglasses
                labels = [(1.0, 0.0), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
                labels= np.array(labels)

                # Decode the latent vectors into images using CVAE
                generated_images = cvae.decode(z, labels).numpy()
                output_folder = 'images_saved'
                
                # Loop through each image in the batch
                for i in range(batch):
                    # Reverse scaling (assuming images were scaled by dividing by 255 before training)
                    image = (generated_images[i] * 255).astype(np.uint8)
                
                    # Display the image
                    plt.figure(figsize=(2, 2))
                    plt.imshow(image)  # No cmap needed for RGB images
                    plt.title(f"Generated Image {i+1}")
                    plt.axis("off")
                    plt.show()
                    file_name = os.path.join(output_folder, f'generated_{i}.png')  # Create file path
                    img = Image.fromarray(image)
                    img.save(file_name)
                
            else:
                # Handle the case where weights_file is None
                logger.error(f"No weights file found for CVAE {exp_name} in chk folder")
                
    
###############################################################################
#### RUN DEF ##################################################################
def run_cvae(data, 
             data_label,
             data_val,
             data_label_val ,
             IMAGE_H,
             IMAGE_W , 
             label_dim,
             latent_dim,
             checkpoint_dir='checkpoints',
             learning_rate=0.001,
             encoder_filters=(32, 16),
             decoder_filters=(16, 32),
             epochs=100,
             batch_size=16, 
             patience=20,
             early_stopping_interval=10):
        
        # Set up strategy based on available GPUs
        strategy = setup_strategy()
       
        # Initialize the CVAE within the strategy scope
        with strategy.scope():
            cvae = t_cvae_class.ConditionalVariationalAutoencoder(
                                                                    IMAGE_H,
                                                                    IMAGE_W ,
                                                                    label_dim,
                                                                    latent_dim, 
                                                                    learning_rate,
                                                                    encoder_filters,
                                                                    decoder_filters)

            strategy.run(lambda: cvae.fit(
                                        data,
                                        data_label,
                                        data_val,
                                        data_label_val,
                                        epochs=epochs, 
                                        batch_size=batch_size,
                                        patience=patience,
                                        early_stopping_interval=early_stopping_interval,
                                        checkpoint_dir=checkpoint_dir))
        return cvae

###############################################################################
    


###############################################################################

def main():
    ### PATH DEFINITION #######################################################
    path_root_data = Path(__file__).resolve().parent
    DATASET_PATH = path_root_data /'lfw-deepfunneled' 
    PICKLE_FILE_IMG = path_root_data/'all_photos.pkl'
    PICKLE_FILE_ATTRS = path_root_data/'all_attrs.pkl'
    TXT_FILE_ATTR = path_root_data/'lfw_attributes.txt'
    JSON_EXPERIMENT = path_root_data/'experiments.json'
    CHECKPOINT_PATH = path_root_data  / 'checkpoint/' # path for model chaekpoint
    ###
    dx=dy = 60 ## cropping
    dimx = dimy = 60 # resizing
    ###
    enable_trainig = False ### if False check for saved model weight
    ### IMAGE PROCESSING STEP ################################################
    # Check if the img pickle file exists
    if os.path.exists(PICKLE_FILE_IMG) and os.path.exists(PICKLE_FILE_ATTRS):
        # Load processed images from the pickle file
        with open(PICKLE_FILE_IMG, 'rb') as f:
            all_photos = pickle.load(f)
        
        with open(PICKLE_FILE_ATTRS, 'rb') as f:
            df_attrs = pickle.load(f)
            
        logger.info("Loaded images from pickle file")
    
    else:
        logger.info("Starting images processing")
        
        all_photos, df_attrs = fetch_dataset(dx=dx,dy=dy,
                                   dimx=dimx,dimy=dimy,
                                   DATASET_PATH=DATASET_PATH,
                                   TXT_FILE_ATTR=TXT_FILE_ATTR)
        
        with open(PICKLE_FILE_IMG, 'wb') as f:
            pickle.dump(all_photos, f)
        with open(PICKLE_FILE_ATTRS, 'wb') as f_a:
             pickle.dump(df_attrs, f_a)
        logger.info("Saved images to pickle file")
    
    # Load the JSON file with vae experiments
    with open(JSON_EXPERIMENT, "r") as json_file:
        experiments = json.load(json_file)
    

    ###########################################################################
    ### GET IMAGES WITH SOME INTERESTING ATTRIBUTES --> SMILING AND BALD
    df_cond = df_attrs[['Sunglasses','Smiling']]
    
    ## RESCALING ####
    data = np.array(all_photos / 255, dtype='float32')
    # Min-Max scaling for each column separately
    df_cond_scaled = df_cond.copy()
    # Scale each column separately
    for column in df_cond_scaled.columns:
        min_val = df_cond_scaled[column].min()
        max_val = df_cond_scaled[column].max()
        df_cond_scaled[column] = (df_cond_scaled[column] - min_val) / (max_val - min_val)
    
    #Round the values to 3 decimal places before converting to NumPy array
    df_cond_scaled_float = df_cond_scaled.astype(np.float32)
    df_cond_scaled_rounded = df_cond_scaled_float.round(3)
    data_cond_scaled = df_cond_scaled_rounded.values
    
    ### TRAIND A VALIDAITON DATASET
    test_size = 0.2  # 20% for validation

    # Split the data into training and validation sets
    data_train, data_val, data_cond_train, data_cond_val = train_test_split(
        data, data_cond_scaled, test_size=test_size, random_state=42
    )
    
    ## start runner

    _ = runner(
                experiments, 
                data_train,
                data_cond_train,
                data_val,
                data_cond_val,
                enable_trainig = enable_trainig,
                path_save_checkpoint=CHECKPOINT_PATH
                )
    
    
    
    
###############################################################################
if __name__ == "__main__":
    main()

from load_data.keras_generators import train_data_gen
from utils import assign
import config
from models.abstract import ModelWrapper

def train_op(model_instance : ModelWrapper, 
             model_name,
             train_dir,
             batch_size, 
             epochs, 
             val_dir = None,
             val_split = 0.4,
             seed = 1234,
             resume = False, 
             backup = False,
             augment = True):
    
    #get a generic logger at INFO level
    logger = config.global_logger
    
    logger.info("\n\n===> {:s} <===\n".format(model_name))

    logger.info("=> Training data will be loaded from {:s}".format(train_dir))
    if not val_dir is None:
        logger.info("=> Validation data will be loaded from {:s}".format(val_dir))
    else:
        logger.info("=> VAL split: {:0.2f}".format(val_split))
        logger.info("=> seed: {:s}\n".format(str(assign(seed, "None"))))

    IMG_HEIGHT, IMG_WIDTH = model_instance.input_shape()
    
    #Set hard-coded hyper-parameters

    
    #set up the image loaders for training and validation

    logger.info("=> BATCH SIZE: {:d}".format(batch_size))
    logger.info("=> Learning Rate {:f}".format(model_instance.l_rate))
    
    
    if augment:
        tr_gen, val_gen = train_data_gen(
            train_dir = train_dir, 
            val_dir = val_dir,
            BATCH_SIZE = batch_size, 
            val_split = val_split, 
            seed = seed, 
            IMG_HEIGHT = IMG_HEIGHT, 
            IMG_WIDTH = IMG_WIDTH, 
            class_mode = 'categorical',
            
            #augmentation
            horizontal_flip = True,
            zoom_range = 0.20,
            shear_range = 0.20,
            width_shift_range = 0.3,
            height_shift_range = 0.3,
            rotation_range = 35,
            )
    else:    
        
        tr_gen, val_gen = train_data_gen(
            train_dir = train_dir, 
            val_dir = val_dir,
            BATCH_SIZE = batch_size, 
            val_split = val_split, 
            seed = seed, 
            IMG_HEIGHT = IMG_HEIGHT, 
            IMG_WIDTH = IMG_WIDTH, 
            class_mode = 'categorical',
            )
    
    #train the model
    model_instance.fit_generator(
            epochs = epochs,
            train_gen = tr_gen, 
            val_gen = val_gen, 
            resume_training = resume, 
            # None means that the weights will be loaded (if any) from the default location determined by 
                # model_name, working_dir and model_name_2
            weight_filename = None, 
            BACKUP_OLD_MODEL_DATA = backup
            )
    
    logger.info("=> **********************************************")
    logger.info("=> Training is over")
    logger.info("=> The model has been save in {:s}".format(model_instance.checkpoint.filename))  
    
    

import argparse
import importlib
import os
import warnings
import torch
import pandas as pd
import pytorch_lightning as pl
import numpy as np
import math
import itertools
import inspect
import numbers

from datetime import date, time, datetime, timedelta
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from tqdm import tqdm 
from pytorch_lightning.tuner import Tuner
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import precision_score, accuracy_score, recall_score

import tud_presence_prediction.helpers.my_logging 
from tud_presence_prediction.data.internal.data_processing import data_processing_util
from tud_presence_prediction.data.internal.global_ts_dataset import global_timeseries_dataset
from tud_presence_prediction.models.internal.model_util import model_util


class presence_prediction:
    """
    This class represents the entry point for all functionality offered by this project. One presence prediction instance should be initialized for one task. 
    
    A presence prediction combines the data provided by a given data procurer with a given Lightning module representing a PyTorch model. 
    These classes must be defined in files within the "model" and "data" directories. A data procurer must be subclass of the data_procurer class. A model must ba a subclass of a LightningModule.
    Each of them must be contained in their own file. The class must have the same name as the file, excluding the file extension.

    When initializing a presence prediction, their names must be given. After initializing, a training, a prediction or an evaluation may be performed.
    The data will be retrieved from the data procurer using standardized loading and processing methods. It will then be given to the model to perform the desired task.
    """

    # default settings for parameters
    default_model_file = "LearnTransformer"    # python file within the 'models' directory, containing a lightning module.
    default_data_procurer_file = "dynamic_multiuser"    # python file within the 'data' directory, containing a data_procurer.
    default_data_from_storage = False                   # wether data should be retrieved from local storage if available.
    default_data_to_storage = False                     # wether data should be saved to local storage.
    default_data_to_readable_file = False               # wether data should be saved in a readable format in addition to the local storage format. Independent from local storage. 
    default_data_storage_file = "auto"                  # local data storage name. If "auto" is given, it will automatically be resolved using datestrings.
    default_logging_mode = "CONSOLE"                    # defines the amount of logging. 'Full' generates text log files and images. 'Console' generates no files. 
    default_worker_count = 0                            # defines how many threads are used for the data loaders.
    default_epochs = 800                                # defines for how many epochs the training should last.
    default_accelerator = "auto"                        # defines which type of processor is used (GPU or CPU)
    default_batch_size = 10                             # defines the amount of sequences which can be processed in parallel during training
    default_sequence_size = [16,]                       # defines the length of one input sequence, given in days
    default_stride_size = 2                             # defines the offset between two samples sequences for one user, given in days
    default_extrapolation = "now"                       # defines the time up to which extrapolation is performed. "now" should always be used for productive predictions, None for all other actions.
    default_data_time_shift = None                      # the shift in future unknown variables. Used for models that depend on time-shifted data to determine their prediction horizon.
    default_find_learning_rate = False                  # attempt to dynamically find an optimal fixed learning rate.
    default_cloud = False

    # developer settings
    shuffle_batch_contents = True
    live_show_graph = False
    live_save_graph = True
    plot_learning_rate = True
    log_timings = False
    deterministic = True
    dynamic_pos_weight_heuristic = True


    def __init__(self, model_file=None, data_procurer_file=None, data_time_shift=None, data_from_storage=None, data_to_storage=True, data_storage_filename="user_data", data_to_readable_file=True, batch_size=None, sequence_size=None, stride_size=None, find_learning_rate=None, version=None, logging_mode=None, user=None, date=None, days=None, user_home_coordinates=None, user_data=None, extrapolation="now", split=20, epochs=None, accelerator=None, workers=None, production=False, model_path_production=None, cloud=None):
        """
        Initializes a new presence prediction for a given model and a given data procurer.

        Args:
            model_file (str):
                The name of the python file defining the desired LightningModule. The module's class name must match the file name. One file may only define one model, which will be used to perform the desired task within this presence prediction.
            data_procurer_file (str):
                The name of the python file defining the desired data_procurer. The procurer's class name must match the file name. One file may only define one data_procurer, which will be used to retrieve the data for the desired task. For predictions, a data procurer returning only a single user should be used. For other task, multiple datasets are supported.
            data_time_shift (str):
                A standardized string defining the desired time-shift. The string must be given in the format 'x:y', where x defines the shift amount in days and y defines the shift amount in timeslots. Wether a time-shift is required depends on the used model. The amount of time-shift defines the trained model's prediction horizon. Currently, only full days may be shifted. Example: If a model should be able to predict 24 hours into the future, a time-shift of 1:0 should be used.
            data_from_storage (bool):
                If True, the presence prediction will attempt to find the most recent local file containing data for the given data procurer. A file must have been created on previous executions using data_to_storage=True.
            data_to_storage (bool):
                If True, the used data procurer will store the loaded raw data in a .txt file to be used for future tasks. This may be used to ensure consistency of data for testing purposes.
            data_storage_filename (str):
                The name of the file to be used when loading data from local storage. If set to None while data_from_storage is True, the newest file matching the given data procurer will automatically be selected. Files must first have been created using data_to_storage=True.
            data_to_readable_file (bool):
                If True, additional CSV files will be created from the data procurer's raw and processed data. These files are independent from storage files and are exclusively intended for manual inspection.
            batch_size (int):
                The size one training batch. This parameter is only used for training. It should be adapted to the used computational accelerator. If additional computing power and VRAM are available, this may be increased to increase training efficiency.
            sequence_size (list of int or int):
                The standardized length of one user input for the model, given in days. This parameter is used for training, evaluation and predictions. Multiple integer values may be given if a model should be able to use multiple lengths. This likely increases the ability to generalize on inputs of different lengths, but also increases training time. If a model version is trained on a specific time-shift, the same time-shift must be given for all actions performed on that version.
            stride_size (int):
                The time between two training samples, given in days. This parameter is only used for training. It defines how much time is skipped between two training samples sequences for one user. Lower values will generate more samples. More samples will increase training time but may improve training results. A recommended default value for most models would be half the smallest sequence size. If a model version is trained on a specific sequence length, the same value(s) must be given for all actions performed on that version.
            find_learning_rate (bool):
                If True, the presence prediction will try to dynamically determine an appropriate learning rate based on the given data and the given model. This feature seems unreliable and at the time of this writing, all current models implement their own learning rate schedulers. It is therefore recommended to keep this feature deactivated.
            version (str):
                The version name of a trained model. Optional for training, mandatory for prediction and evluation. Default version names are incremental numbers generated by Lightning after each training. Each training generates a new version directory. When then using this parameter, a checkpoint from within the models training result directory matching the given version name is selected dynamically.
            logging_mode (str):
                Defines the extent of used logging. Available values are "none", "console", "text" and "full". The setting "full" will allow image generation and all other types of logging. The setting "text" will allow text file generation and console logging. The setting "console" will only use console outputs.
            user (int):
                This must be a value uniquely identifying a user. This value is forwarded to the data procurer and used when loading data. Not all data procurers must use this value. At the time of this writing single user data procurers use internal IDs to identify users, get additional information from a local user collection and then load their data via HTTP.
            date (date):
                The date a test prediction should be performed for. Important: This is only intended to be used for test predictions for past dates. For all productive predictions, the current date will be selected automatically and extrapolation should be used to ensure that a prediction always begins at the current point in time.
            days (int):
                The length of a prediction, given in days. This parameter is only used for predictions and directly defines the size of the generated output of the prediction function. A prediction is always standardized to begin at the current day and yield a total amount of days equal to this parameter. If this is set to 1, then only the remaining predictions for the current day are returend. Any length may be given, but the quality of the results depends strongly on the used model. A time-shift model will only yield reliable results within it's trained time-shift window. A regressive model will yield gradually decreasing accuracies over longer windows. 
            user_home_coordinates (str):
                A users home coordinates, given in a format known to the chosen data procurer. This information is forwarded to the data procurer. Not all data procurers use this information. At the time of this writing, this is only used by one data procurer to procure a single user's data. This data procurer may either retrieve data from a HTTP source or use the contents of this parameter if provided to use locally available data. It effectively replaces the data procurer's loading process by replacing a HTTP request with directly available data given here.
            user_data (str):
                A users relative location data, given in a format known to the chosen data procurer. This information is forwarded to the data procurer. Not all data procurers use this information. At the time of this writing, this is only used by one data procurer to procure a single user's data. This data procurer may either retrieve data from a HTTP source or use the contents of this parameter, if provided, to use locally available data. It effectively replaces the data procurer's loading process by replacing a HTTP request with directly available data given here.
            extrapolation (str):
                Defines the amount of used extrapolation. This information is forwarded to the data procurer. Available values are "none", "last" and "now". As "now" will extrapolate data to the current time, it is mandatory to be used for productive predictions. A model is trained to predict the continuation of a users locations based on previous locations. It is therefore important that every input to the model ends at the timestamp where a prediction is desired to begin. For all other purposes, the option "none" should be used.
            split (int):
                The percentual amount of the procured data that should be used for validation, given in values from 0 to 100. This information is forwarded to the data procurer. It is intended to be used for training and evaluation. If used, the same split should be used for training and evaluation.
            epochs (int):
                The amount of epochs the training is intended to last for. 
            accelerator (str):
                The name of the desired accelerator. Supported values are "auto", "cpu", "cuda" and other common devices. Defaults to "auto", which will select cuda if available, otherwise it will fall back to CPU. Used for prediction, training and evaluation.
            workers (int):
                The amount of workers to be used for the DataLoaders during training time. When set to 0, the DataLoaders will work within the main thread, which is effective for low-cost batch generation. When computationally intensive tasks must be performed to generate one batch, it may be beneficial to increase this value. At the time of this writing, generating batches is trivial and a value of 0 is optimal.
            production (bool):
                If true, the model will be loaded from the model_production_path and not from the internal training_results folder inside the repository
            model_path_production (str):
                The model path from which the model needs to be loaded in production 
            cloud (bool):
                If true, the logging directory will be the one of GCP. Otherwise, the logging directory will be the local directory
        """
        
        # prepare all params
        if data_from_storage == True and data_to_storage is None: 
            self.data_to_storage = False
            self.data_from_storage = True
        if data_to_storage == True and data_from_storage is None:
            self.data_to_storage = True
            self.data_from_storage = False
        else:
            self.data_from_storage = data_from_storage if data_from_storage is not None else self.default_data_from_storage #(False if data_to_storage == True else presence_prediction.default_data_from_storage)
            self.data_to_storage = data_to_storage if data_to_storage is not None else self.default_data_to_storage #(False if data_from_storage == True else presence_prediction.default_data_to_storage)

        if type(sequence_size) == int: sequence_size = [int(sequence_size),]
        elif type(sequence_size) == tuple: sequence_size = list(sequence_size)
        

        self.data_procurer_file = data_procurer_file if data_procurer_file is not None else self.default_data_procurer_file
        self.model_file = model_file if model_file is not None else self.default_model_file
        self.batch_size = batch_size if batch_size is not None else self.default_batch_size
        self.sequence_size = sequence_size if sequence_size is not None else self.default_sequence_size
        self.stride_size = stride_size if stride_size is not None else self.default_stride_size
        self.data_time_shift = data_time_shift if data_time_shift is not None else self.default_data_time_shift
        self.epochs = epochs if epochs is not None else self.default_epochs
        self.find_learning_rate = find_learning_rate if find_learning_rate is not None else self.default_find_learning_rate
        self.data_to_readable_file = data_to_readable_file if data_to_readable_file is not None else self.default_data_to_readable_file
        self.data_storage_file = data_storage_filename if data_storage_filename is not None else self.default_data_storage_file
        self.logging_mode = logging_mode if logging_mode is not None else self.default_logging_mode
        self.cloud = cloud if cloud else self.default_cloud
        self.log_base_directory = os.path.join("tud_presence_prediction","training_results", self.model_file)
        self.workers = workers if workers is not None else self.default_worker_count
        self.accelerator = str.lower(accelerator) if accelerator is not None else self.default_accelerator
        self.extrapolation = extrapolation if extrapolation is not None else self.default_extrapolation
        self.date = date
        self.days = days
        self.user = user
        self.user_home_coordinates = user_home_coordinates
        self.user_data = user_data
        self.split = split
        self.version = version
        self.datasets = None
        self.sequence_size.sort(reverse=True)
        self.production = production
        self.model_path_production = model_path_production
        
        self.lightning_profiler_mode = "simple" if self.log_timings else None

        # initialize logger
        self.logger = tud_presence_prediction.helpers.my_logging.logger(self.logging_mode, use_time_profiler=self.log_timings)

        # set up checkpoints
        checkpoints = []
        tl_checkpoint_callback = ModelCheckpoint(
            filename="training-{epoch:0>7}-best_loss_ckpt-{train_loss:.2f}",
            every_n_epochs=1,
            monitor='train_loss', 
            mode='min'
        ) 
        ta_checkpoint_callback = ModelCheckpoint(
            filename="training-{epoch:0>7}-best_acc_ckpt-{train_accuracy:.2f}",
            every_n_epochs=1,
            monitor='train_accuracy', 
            mode='max'
        ) 
        tp_checkpoint_callback = ModelCheckpoint(
            filename="training-{epoch:0>7}-best_prec_ckpt-{train_precision:.2f}",
            every_n_epochs=1,
            monitor='train_precision', 
            mode='max'
        ) 
        checkpoints.extend((tl_checkpoint_callback, ta_checkpoint_callback, tp_checkpoint_callback))
        if self.split != None and float(self.split) > 0:
            vl_checkpoint_callback = ModelCheckpoint(
                filename="validation-{epoch:0>7}-best_loss_ckpt-{val_loss:.2f}",
                every_n_epochs=1,
                monitor='val_loss', 
                mode='min'
            ) 
            va_checkpoint_callback = ModelCheckpoint(
                filename="validation-{epoch:0>7}-best_acc_ckpt-{val_accuracy:.2f}",
                every_n_epochs=1,
                monitor='val_accuracy', 
                mode='max'
            ) 
            vp_checkpoint_callback = ModelCheckpoint(
                filename="validation-{epoch:0>7}-best_prec_ckpt-{val_precision:.2f}",
                every_n_epochs=1,
                monitor='val_precision', 
                mode='max'
            )
            checkpoints.extend((vl_checkpoint_callback, va_checkpoint_callback, vp_checkpoint_callback))
        last_checkpoint_callback = ModelCheckpoint(
            filename="conclusion-{epoch:0>7}",
            every_n_epochs=1,
            monitor=None
        ) 
        checkpoints.append(last_checkpoint_callback)

        # ensure reproducability of results
        if self.deterministic == True: pl.seed_everything(2, workers=True)

        # set up logging for LR
        lr_monitor = LearningRateMonitor(logging_interval='step')

        # initialize trainer (done early for logging and device selection)
        if not os.path.exists(self.log_base_directory) and not production: os.makedirs(self.log_base_directory)
        self.trainer = pl.Trainer(profiler=self.lightning_profiler_mode, enable_progress_bar=True, deterministic=self.deterministic, max_epochs=self.epochs + 1, log_every_n_steps=1, accelerator=self.accelerator, devices="auto", default_root_dir=self.log_base_directory, callbacks=[*checkpoints, self.logger, lr_monitor], logger=CSVLogger(save_dir=self.log_base_directory), num_sanity_val_steps=-1)
        self.log_directory = self.trainer.logger.log_dir
        self.logger.set_experiment_info(self.log_directory, self.trainer.logger.version, self.model_file, self.data_procurer_file, self.data_from_storage)
        
        # determine used accelerator
        self.used_accelerator = str(type(self.trainer.accelerator).__name__).lower().replace("accelerator", "")
        if "cuda" in self.used_accelerator.lower():
            self.used_precision = "medium"
        else:
            self.used_precision = "highest"
        if not self.cloud:
            torch.set_float32_matmul_precision(self.used_precision) # torch recommends this

        # log all given information
        self.logger.headline("Initializing new presence prediction")
        self.logger.info("Used options for presence predition: ")
        self.logger.info(f"File for lightning model definition: '{self.model_file}'" + (" was given. " if model_file is not None else " was used by default."))
        self.logger.info(f"File for data procurement: '{self.data_procurer_file}'" + (" was given. " if data_procurer_file is not None else " was used by default."))
        self.logger.info(f"User parameter for data procurement: '{self.user}'" + (" was given. " if user is not None else " was used by default."))
        self.logger.info(f"Date parameter for data procurement: '{self.date}'" + (" was given. " if date is not None else " was used by default."))
        self.logger.info(f"Days parameter for data procurement: '{self.days}'" + (" was given. " if days is not None else " was used by default."))
        self.logger.info(f"User home coordinates for data procurement: coordinates" + (" were given. " if user_home_coordinates is not None else " were not given."))
        self.logger.info(f"User data for data procurement: data" + (" was given. " if user_data is not None else " was not given."))
        self.logger.info(f"Extrapolate parameter for data procurement: '{self.extrapolation}'" + (" was given. " if extrapolation is not None else " was used by default."))
        self.logger.info(f"Split parameter for data procurement: '{self.split}'" + (" was given. " if split is not None else " was used by default."))
        self.logger.info(f"Batch size: '{self.batch_size}'" + (" was given. " if batch_size is not None else " was used by default."))
        self.logger.info(f"Sequence length: '{self.sequence_size}'" + (" was given. " if sequence_size is not None else " was used by default."))
        self.logger.info(f"Time shift: '{self.data_time_shift}'" + (" was given. " if data_time_shift is not None else " was used by default."))
        self.logger.info(f"Gathering data locally: '{self.data_from_storage}'" + (" was given. " if data_from_storage is not None else " was used by default."))
        self.logger.info(f"Storing data locally: '{self.data_to_storage}'" + (" was given. " if data_to_storage is not None else " was used by default."))
        self.logger.info(f"Local data storage file: '{self.data_storage_file}'" + (" was given. " if data_storage_filename is not None else " was used by default."))
        self.logger.info(f"Storing data in readable format: '{self.data_to_readable_file}'" + (" was given. " if data_to_readable_file is not None else " was used by default."))
        self.logger.info(f"Logging mode: '{self.logging_mode}'" + (" was given. " if logging_mode is not None else " was used by default."))
        self.logger.info(f"Amount of workers for DL: '{self.workers}'" + (" was given. " if workers is not None else " was used by default."))
        self.logger.info(f"Given accelerator: '{self.accelerator}'" + (" was given. " if accelerator is not None else " was used by default."))
        self.logger.info(f"Used accelerator: '{self.used_accelerator}'" + (" determined automatically."))
        self.logger.info(f"Used precision: '{self.used_precision}'" + (" was determined based on the used accelerator. The setting 'medium' is recommended for CUDA-based accelerators."))
        self.logger.info(f"Find learning rate dynamically: '{self.find_learning_rate}'" + (" was given. " if find_learning_rate is not None else " was used by default."))
        self.logger.info(f"Number of epochs for training: '{self.epochs}'" + (" was given. " if epochs is not None else " was used by default."))
        self.logger.info(f"Version: '{self.version}'" + (" was given. " if version is not None else " was given, starting with fresh model."))
        self.logger.info("Cloud mode" if self.cloud else "Local mode")

        # initialize data procurer
        self.logger.headline(f"Loading data procurer '{self.data_procurer_file}'")
        self.data_procurer_name = self.data_procurer_file
        self.data_procurer_subclass = getattr(importlib.import_module(".data." + self.data_procurer_file, package="tud_presence_prediction"), self.data_procurer_name) # TODO: do additional sanity checks (e.g. see if file is there and maybe in correct directory)
        self.data_procurer = self.data_procurer_subclass(
            logger = self.logger,
            from_storage = self.data_from_storage,
            to_storage = self.data_to_storage,
            storage_file = self.data_storage_file,
            store_readable = self.data_to_readable_file
        ) 
        self.logger.info(f"Data procurer {self.data_procurer.__class__.__name__} loaded succesfully.")

        # load data
        self.datasets = self.data_procurer.load(self.data_time_shift, date=self.date, days=self.days, user=self.user, user_home_coordinates=self.user_home_coordinates, user_data=self.user_data, split=self.split, extra=self.data_time_shift, extrapolate=self.extrapolation)


        # initialize model class
        self.logger.headline(f"Loading model {self.model_file}")
        self.model_name = self.model_file
        self.model_module_class = getattr(importlib.import_module(".models." + self.model_file, package="tud_presence_prediction"), self.model_name)

        # initialize fresh model
        if self.version == None:
            feature_dimensions = [input_tensor.size(-1) for input_tensor in self.datasets[0][0].tensors[0:-1]]
            
            if len(feature_dimensions) < 3: feature_dimensions.extend([None for each_ in range(0, 3 - len(feature_dimensions))])

            self.model = self.model_module_class(*feature_dimensions)
            print("Derived feature dimensions:", feature_dimensions)
            self.logger.info(f"Model {self.model.__class__.__name__} loaded successfully.")
        # load model from checkpoint
        elif self.version != None and not self.production:
            self.version_directory_path = os.path.join(os.path.expanduser("~/scratch"), "presence_prediction", "tud_presence_prediction", "training_results", self.model_file, "lightning_logs", "version_" + str(self.version), "checkpoints") # potentially replace this with more adjustable path
            
            all_ckpt_files = [filename for filename in os.listdir(self.version_directory_path) if filename.endswith(".ckpt")]
            all_ckpt_files.sort()
            
            if len(all_ckpt_files) <= 0:
                raise ValueError(f"No trained model could be found at: {self.version_directory_path}.")
            
            found_filename = all_ckpt_files[-1]
            self.full_ckpt_path = os.path.join(self.version_directory_path, found_filename)
            self.logger.info(f"Loading from version {self.version}, checkpoint '{found_filename}'.")

            # load previously trained state for the model
            self.model = self.model_module_class.load_from_checkpoint(self.full_ckpt_path) # may be adjust to fit differently shaped data by first inspecting the dataset and then passing params, e.g. in_dim=128, out_dim=10
            self.model_stats = torch.load(self.full_ckpt_path)
            self.logger.set_starting_epoch(self.model_stats['epoch'])
            self.logger.info(f"Model loaded from {self.model_stats['epoch']} epochs of training.")
        # load model in production
        elif self.model_path_production != None and self.production:
            # load previously trained state for the model
            self.model = self.model_module_class.load_from_checkpoint(self.model_path_production) # may be adjust to fit differently shaped data by first inspecting the dataset and then passing params, e.g. in_dim=128, out_dim=10
            if "cuda" in self.used_accelerator.lower():
                self.model_stats = torch.load(self.model_path_production)
            else:
                self.model_stats = torch.load(self.model_path_production, map_location=torch.device('cpu'))
            self.logger.set_starting_epoch(self.model_stats['epoch'])
            self.logger.info(f"Model loaded from {self.model_stats['epoch']} epochs of training.")

        # ignore known warnings
        warnings.filterwarnings("ignore", ".*does not have many workers.*")
        warnings.filterwarnings("ignore", ".*Experiment logs directory.*")
        warnings.filterwarnings("ignore", ".*You defined a `validation_step` but have no `val_dataloader`.*")
        warnings.filterwarnings("ignore", ".*mean of empty slice.*")
        

    
    def train(self):
        """Performs the model training based on the initially given parameters."""

        self.logger.headline("Starting Training")
        self.logger.info(f"Storing results in version {self.trainer.logger.version}.")
        self.logger.info(f"Training on {len(self.datasets)} users.")
        self.logger.info(f"Generating sequences of length {self.sequence_size}.")
        self.logger.info(f"Generating subsequent sequences with {self.stride_size} days inbetween.")
        self.logger.info(f"Processing up to {self.batch_size} sequences in parallel.")

        if self.extrapolation != "none":
            self.logger.critical("Warning: Using data extrapolation up to current datetime for training purposes. This may heavily alter training data if users are present whose data is significantly older.") 

        train_datasets = []
        val_datasets = []

        global_train_datasets = []
        global_val_datasets = []

        train_dataloaders = []
        val_dataloaders = []

        train_variation_counts = []
        val_variation_counts = []

        train_padding_counts = []
        val_padding_counts = []

        sanity_check_train_variations = 0
        sanity_check_val_variations = 0

        train_batch_count = 0
        val_batch_count = 0

        train_average_dataset_metrics = dict()
        val_average_dataset_metrics = dict()

        # divide data into lists of training and validation data
        for user_index in range(len(self.datasets)):
            for dataset_type in range(len(self.datasets[user_index])):
                datasets_array = train_datasets if dataset_type == 0 else val_datasets
                datasets_array.append(self.datasets[user_index][dataset_type])

        # create global datasets. One dataset per desired window size and per set type (training, validation)
        for window_index, window_size in enumerate(self.sequence_size):
            for dataset_type in range (0,2):
                global_dataset_list = global_train_datasets if dataset_type == 0 else global_val_datasets
                dataloader_list = train_dataloaders if dataset_type == 0 else val_dataloaders
                ds_array = train_datasets if dataset_type == 0 else val_datasets

                new_global_dataset = None

                # add global dataset to final list
                next_size = self.sequence_size[window_index + 1] if window_index + 1 < len(self.sequence_size) else None
                if next_size != None:
                    # add a limit for padding so that data goes into the smallest fitting category
                    new_global_dataset = global_timeseries_dataset(ds_array, window_size=window_size, stride=self.stride_size, window_size_compensation="padding", window_size_compensation_limit=next_size)
                else:
                    # the smallest category has no upper bounds
                    new_global_dataset = global_timeseries_dataset(ds_array, window_size=window_size, stride=self.stride_size, window_size_compensation="padding")

                if len(new_global_dataset) > 0:
                    use_shuffle = self.shuffle_batch_contents if dataset_type == 0 else False # only use shuffle for training DataLoaders
                    global_dataset_list.append(new_global_dataset)
                    dataloader_list.append(DataLoader(new_global_dataset, batch_size=self.batch_size, shuffle=use_shuffle, num_workers=self.workers))

        # get the variation counts and paddings lengths which the datasets calculated internally
        train_variation_counts = [global_train_set.dataset_variations for global_train_set in global_train_datasets]
        val_variation_counts = [global_val_set.dataset_variations for global_val_set in global_val_datasets]
        total_train_variation_count = sum([len(global_train_ds) for global_train_ds in global_train_datasets])
        total_val_variation_count = sum([len(global_val_ds) for global_val_ds in global_val_datasets])
        train_padding_counts = [global_train_set.dataset_paddings for global_train_set in global_train_datasets]
        val_padding_counts = [global_val_set.dataset_paddings for global_val_set in global_val_datasets]

        # generate and log user data metrics, metric averages and sanity check values
        for user_index in range(len(self.datasets)):
            for dataset_type in range(len(self.datasets[user_index])):
                if self.datasets[user_index][dataset_type] == None: continue
                set_name = "Training" if dataset_type == 0 else "Validation" 

                variation_list = train_variation_counts if dataset_type == 0 else val_variation_counts
                user_variation_count = sum([size_variation_list[user_index] for size_variation_list in variation_list])
                padding_list = train_padding_counts if dataset_type == 0 else val_padding_counts
                user_padding_count = sum([padding_list[user_index] for padding_list in padding_list])

                dataset_metrics = data_processing_util.generate_dataset_metrics(self.datasets[user_index][dataset_type])
                self.logger.info(f"{set_name} set metrics for user {user_index}:", True) 
                self.logger.info(f"- {dataset_metrics['samples']} days with {user_variation_count} variations{(' (with ' + str(user_padding_count) + ' generated padding)') if user_padding_count > 0 else ''}.") 
                self.logger.info(f"- {round((dataset_metrics['labeled_count']/dataset_metrics['total_label_fields_count']) * 100)}% of all data is labeled with ~{round(dataset_metrics['interpolated_percentage']*100, 1)}% interpolation.")
                self.logger.info(f"- {round((dataset_metrics['positive_count']/dataset_metrics['labeled_count']) * 100)}% of labeled data has label 1.")
                dataset_metrics["variations"] = user_variation_count

                if dataset_type == 0:
                    sanity_check_train_variations += user_variation_count
                elif dataset_type == 1: 
                    sanity_check_val_variations += user_variation_count

                datasets_array = train_datasets if dataset_type == 0 else val_datasets
                datasets_array.append(self.datasets[user_index][dataset_type])
                average_metrics = train_average_dataset_metrics if dataset_type == 0 else val_average_dataset_metrics
                for metric in dataset_metrics:  
                    if isinstance(dataset_metrics[metric], numbers.Number): average_metrics[metric] = (dataset_metrics[metric] * user_variation_count) + average_metrics.get(metric, 0)

            if len(self.datasets[user_index]) < 2 or self.datasets[user_index][1] == None: self.logger.info(f"No validation data set has been provided for user {user_index}.") 

        # get dataset metric averages across all users
        for metric in train_average_dataset_metrics:
            train_average_dataset_metrics[metric] = train_average_dataset_metrics[metric] / total_train_variation_count
        for metric in val_average_dataset_metrics:
            val_average_dataset_metrics[metric] = val_average_dataset_metrics[metric] / total_val_variation_count

        # log average data metrics
        self.logger.info(f"Average training set metrics across all users:", True) 
        self.logger.info(f"- {round(train_average_dataset_metrics['samples'], 1)} samples per user with with {round(train_average_dataset_metrics['variations'], 1)} total variations.") 
        self.logger.info(f"- {round((train_average_dataset_metrics['labeled_count']/train_average_dataset_metrics['total_label_fields_count']) * 100)}% of all data is labeled with ~{round(train_average_dataset_metrics['interpolated_percentage']*100, 1)}% interpolation.")
        self.logger.info(f"- {round((train_average_dataset_metrics['positive_count']/train_average_dataset_metrics['labeled_count']) * 100)}% of labeled data has label 1.")

        if total_val_variation_count > 1:
            self.logger.info(f"Average validation set metrics across all users:", True) 
            self.logger.info(f"- {round(val_average_dataset_metrics['samples'], 1)} samples per user with {round(val_average_dataset_metrics['variations'], 1)} total variations.") 
            self.logger.info(f"- {round((val_average_dataset_metrics['labeled_count']/val_average_dataset_metrics['total_label_fields_count']) * 100)}% of all data is labeled with ~{round(val_average_dataset_metrics['interpolated_percentage']*100, 1)}% interpolation.")
            self.logger.info(f"- {round((val_average_dataset_metrics['positive_count']/val_average_dataset_metrics['labeled_count']) * 100)}% of labeled data has label 1.")

        # sanity check variation counts
        assert total_train_variation_count == sanity_check_train_variations, "Training variation count does not match."
        assert total_val_variation_count == sanity_check_val_variations, "Validation variation count does not match."

        # calculate, sanity check and log batch counts
        train_batch_count_projection = math.ceil(sum([sum(train_variation_count_i) for train_variation_count_i in train_variation_counts])/self.batch_size)
        val_batch_count_projection = math.ceil(sum([sum(val_variation_count_i) for val_variation_count_i in val_variation_counts])/self.batch_size)
        train_batch_count = sum([len(train_dl) for train_dl in train_dataloaders])
        val_batch_count = sum([len(val_dl) for val_dl in val_dataloaders])
        assert train_batch_count == train_batch_count_projection, "Unexpected training batch count."
        assert val_batch_count == val_batch_count_projection, "Unexpected validation batch count."
        self.logger.info(f"Training on {train_batch_count} unique batches per epoch for a total of {total_train_variation_count} variations. Last batch sizes per window size: {[len(global_train_ds) % self.batch_size for global_train_ds in global_train_datasets]}", True)
        self.logger.info(f"Validating on {val_batch_count} unique batches per epoch for a total of {total_val_variation_count} variations. Last batch sizes per window size: {[len(global_val_ds) % self.batch_size for global_val_ds in global_val_datasets]}")
        
        # setup DataLoaders to be compatible with lightning's training loop
        multi_dataloader_setup = (len(train_dataloaders) > 1) or (len(val_dataloaders) > 1)
        if multi_dataloader_setup == True:
            self.logger.info("Using multiple DataLoaders.")
            if self.shuffle_batch_contents == True:
                # TODO: Find solution to allow for randomized batches when using multiple DataLoaders
                self.logger.critical("Shuffle is activated but will not be used when training on multiple sequence lengths.")

            # create final data loaders by creating iterators around the newely created ones
            train_dataloaders.append([0])  # workaround for lightning. Creates a placeholder element in the infinite iteration cycle of dataloaders since lightning apparently iterates an additional time at the end of every epoch. Only seems to apply to the training loop.
            self.train_dataloader = itertools.cycle(itertools.chain(*train_dataloaders)) # this workaround is required until lightning implements support for sequential loading within CombinedDataLoaders
            self.val_dataloader = None
            if val_batch_count > 0: self.val_dataloader = itertools.cycle(itertools.chain(*val_dataloaders)) 
            
            # Configure trainer to handle correct amount of batches per epoch. Every unique batch should be handled once per epoch. This is neccessary when using an iterable of DataLoaders.
            self.trainer.limit_train_batches = train_batch_count
            if val_batch_count > 0: self.trainer.limit_val_batches = val_batch_count
        else:
            self.logger.info("Using one DataLoader.")
            self.train_dataloader = train_dataloaders[0]
            self.val_dataloader = None
            if val_batch_count > 0: self.val_dataloader = val_dataloaders[0]

        # setup model util to handle logging and LR scheduling
        if val_batch_count > 0: 
            model_util.enable_loop(self.model, "val")        
        model_util.set_batch_count(train_batch_count)

        # find appropriate learning rate dynamically
        if self.find_learning_rate:
            self.logger.info("Trying to find learning rate dynamically.")
            if "fixed_learning_rate" not in inspect.signature(self.model.__init__).parameters:
                self.logger.info("Model does not support dynamic initialization of learning rate.")
            else:
                tuner = Tuner(self.trainer)
                lr_finder = tuner.lr_find(self.model, train_dataloaders=self.train_dataloader, mode="exponential", num_training=500, attr_name="fixed_learning_rate")

                if self.plot_learning_rate:
                    fig = lr_finder.plot(suggest=True)
                    fig.show()

                new_lr = lr_finder.suggestion()
                self.logger.info(f"Using estimated optimal learning rate: {new_lr}.")
                self.model.hparams.lr = new_lr
                self.model.learning_rate = new_lr

                # reset visual metrics generated by lr_finder
                if hasattr(self.model, 'visual_metrics'):
                    for metric_key_a in self.model.visual_metrics: self.model.visual_metrics[metric_key_a] = []
                    for metric_key_b in self.logger.extra_sanity_data: self.logger.extra_sanity_data[metric_key_b] = []

        # automatically set pos_weights based on distribution of labels. This may speed up training significantly and may also improve final convergence in some cases.
        if self.dynamic_pos_weight_heuristic == True and model_util.get_model_custom_pos_weights_support(self.model):
            pos_fraction = train_average_dataset_metrics['positive_count']/train_average_dataset_metrics['labeled_count']
            neg_fraction = 1 - pos_fraction
            imbalance_ratio = abs((neg_fraction - 0.5) / 0.5)

            # This is a heuristic based on various test cases. More tests with different data may reveal more consise approaches.
            # Ensures that stronger imbalances are not fully compensated, as that may cause an increasingly unstable loss. Small imbalances are compensated almost completely.
            # Alternative viable approach might be to ensure that the label that has less representation is ensured to be be exactly at 0.33 representation after compensation.
            pos_weights = (neg_fraction/pos_fraction) * (1 - imbalance_ratio**2)
            pos_weights = int(pos_weights * 100) / 100 # round down to 2 decimals by truncating
                                     
            model_util.set_model_custom_pos_weights(self.model, pos_weights)
            self.logger.info(f"Weights for positive predictions have automatically been set to {pos_weights}.")
        else:
            hard_pos_weights = getattr(self.model, 'pos_weight', "Unkown")
            if hard_pos_weights != "Unkown": hard_pos_weights = hard_pos_weights.item()
            self.logger.info(f"Dynamic adjustment of label weights is deactivated or not supported by the model. Used weights: {hard_pos_weights}.")
            
        # start visual logging
        self.logger.start_training_graph(self.model, self.epochs, None, self.live_show_graph, self.live_save_graph, self.cloud)

        # printing dataloaders for debugging purposes
        print("train_dataloader:", self.train_dataloader)
        # Zeige die ersten 2 Batches des Trainingsdataloaders
        try:
            print("Erste Trainingsbatches:")
            for i, batch in enumerate(self.train_dataloader):
                print(f"Batch {i}:")
                if isinstance(batch, (list, tuple)):
                    for j, entry in enumerate(batch):
                        print(f"  Entry {j} type: {type(entry)}, shape: {getattr(entry, 'shape', None)}")
                else:
                    print(f"  Batch type: {type(batch)}, value: {batch}")
                if i >= 1:
                    break
        except Exception as e:
            print("Fehler beim Auslesen des Trainingsdataloaders:", e)

        if self.val_dataloader is not None:
            print("val_dataloader:", self.val_dataloader)
            try:
                print("Erste Validierungsbatches:")
                for i, batch in enumerate(self.val_dataloader):
                    print(f"Batch {i}:")
                    if isinstance(batch, (list, tuple)):
                        for j, entry in enumerate(batch):
                            print(f"  Entry {j} type: {type(entry)}, shape: {getattr(entry, 'shape', None)}")
                    else:
                        print(f"  Batch type: {type(batch)}, value: {batch}")
                    if i >= 1:
                        break
            except Exception as e:
                print("Fehler beim Auslesen des Validierungsdataloaders:", e)
        # up until here 

        # fit model
        # train_dataloader is of shape (batch_size, sequence_length, location_dim)
        # the batch size describes how many different sequence lengths are being used in the training loop        
        self.trainer.fit(self.model, train_dataloaders=self.train_dataloader, val_dataloaders=self.val_dataloader) 


        # save training results in pure torch format
        torch.save(self.model.state_dict(), os.path.join(self.log_directory, "checkpoints", "final_torch_states.pth"))
        torch.save(self.model.optimizers().optimizer.state_dict(), os.path.join(self.log_directory, "checkpoints", "final_torch_optimizer.pth"))

        self.logger.info(f"Training ended.")
        self.logger.show_training_graph_permanently()

        #upload trainig results to GCP
        if self.cloud:
            from tud_presence_prediction.helpers.cloud import Cloud
            cloud_directory = Cloud.get_bucket_path()
            self.logger.info(f"Starting uploading model checkpoits annd logs to {cloud_directory} ")
            Cloud.upload_folder(self.log_directory,self.model_file,cloud_directory,self.logger)

        return self.logger.logging_directory


    def evaluate(self, window_size_step=16, window_offset_step=1, prediction_length_step=1, prediction_length_max=5, prediction_settings={"post_processing": "binary percentage cutoff", "post_processing_threshold": 0.5, "adjust_data_length": True, "prioritize_padding": True}, force_mask=False, evaluation_split="training", equalize_user_representation=True, user_treshold=0.65, verbalize_results=True, show_user_presence_prediction=None, generate_recall=False, graph_types=(1,2,3), draw_vertical_contours=True, draw_horizontal_contours=False, use_colormap=True, use_background=False, save_video=True, save_image=True, show_live=True):
        """ 
        Evaluates a trained models performance by varying inputs across multiple dimensions while performing predicitons. Generates multiple 3D graphs to visualize the dependency of accuarcy and precision upon two input dimensions at a time.

        Args:
            window_size_step (int): 
                Desired increase in input sequence length between evaluation steps in days. Keeps increasing until the maximum set by the sequence length parameter is reached. The same sequence length should be used for training and evaluation.
            window_offset_step (int):
                Desired increase in the starting position of each test between evaluation steps in days. Keeps increasing until the smallest user's dataset is exhausted. This is most useful if the model is evaluated on the exact same data it has been trained on. The data procurere's offline storage feature can be used to ensure parity. 
            prediction_length_step (int): 
                Desired increase in the prediction length of each tested batch between evaluation steps in days. Keeps increasing until prediction_length_max is reached. Important: For shift-based models the predictable time span is fixed. This function must be called with the correct parameters to ensure validity of results. If a model has been trained on a 1-day time shift, then the evaluation must have a maximum prediction length of 1 and a prediction length step size of 1.
            prediction_length_max (int):
                The desired maximum prediction length in days. Important: For shift-based models the predictable time span is fixed. This function must be called with the correct parameters to ensure validity of results. If a model has been trained on a 1-day time shift, then the evaluation must have a maximum prediction length of 1 and a prediction length step size of 1.
            prediction_settings (dict):
                A dict containing settings to be passed to the prediction function used internally for every test case. Refer to 'presence_prediction.predict()' to see a list of relevant settings.
            force_mask (string):
                Tries to force a certain type of masking on models that use masks and offer an option to change it. The valid values depend on the used model. A None-type value will force the model to use no masks. Setting this to False will not cause any changes and the model will use it's default value (recommended).
            evaluation_split (str):
                A string containing the type of dataset that should be evaluated. Supported values are "training", "validation" and "training and validation". Splits are determined by the split parameter for the data procurer.
            equalize_user_representation (bool):
                If set to True, the maximum number of test cases is determined by the size of the smallest dataset, thereby ensuring that all datasets have equal representation since all test cases are generated for all datasets. If set to False, all test cases suitable for the largest dataset are used, thereby maximizing the amount of test cases but generating an inconsistent amount of results for each dataset as tests on larger offsets can only be performed on the largest user.
            user_treshold (float): 
                The amount of data a dataset must have to be used for evaluation. Accepts values from 0 to 1 which represent the fraction of the procured data's average length in days. If the average user has a length of 100 days and this is set to 0.5, then user data with a length of less than 50 will be ignored.
            verbalize_results (bool):
                If set to True, results for the most meaninungful test cases are generated in text form in addition to graphical results. 
            show_user_presence_prediction (tuple[int]):
                If set to None, user presence predictions and labels will not be tracked on a individual basis. If set to a tuple containing two integers, prediction results for a window size and a prediction length matching these integers will be tracked per user and shown visually after completion of the test results. This may generate a very large amount of graphs if a large amount of users are being processed. 
            generate_recall (bool):
                If set to True, additional recall values are generated within the accuracy graphs.
            graph_types (list[int]):
                Determines which graphs should be drawn after evaluation. Any combination of the numbers 1, 2 and 3 is valid, order is ignored.
            draw_vertical_contours (bool):
                Determines whether vertical contours should be drawn onto the bottom of the graph, reflecting the vertical axis's values in 2D to potentially improve readablity and aesthetics. 
            draw_vertical_contours (bool):
                Determines whether horizontal contours should be drawn onto the bottom of the graph, reflecting either horizontal axis's values in 2D to potentially improve readablity.
            use_colormap (bool):
                Determines wether a colormap is used to color individual surfaces depending on their accuracy/precision. If false, then a simple directional shading is used to improve readability.
            use_background (bool):
                If set to True, an additional dark background is drawn behind all relevant graphs.
            save_video (bool):
                Determines if a video should be saved to the 'evaluations' directory. If true, then the video saving process starts after the live graph has been closed, if it was generated. This process may take several minutes.
            save_image(bool):
                Determines if a image should be saved to the 'evaluations' directory. If true, then the video saving process starts after the live graph has been closed, if it was generated. 
            show_live (bool):
                Determines if the graph is shown in an interactive window immediatly after completion of the evaluation.
                
                
        """
        import matplotlib.pyplot as plt
        from matplotlib import cbook, cm
        from matplotlib.colors import LightSource, Normalize, ListedColormap
        from matplotlib import animation

        # sanity check extrapolation setting
        if self.extrapolation != "none":
            self.logger.critical("Warning: Using data extrapolation up to current datetime for evaluation purposes. This may heavily alter evaluation data if users are present whose data is significantly older.") 

        # active text logging to save logs 
        evaluation_name = self.logger.get_experiment_string(file_friendly=True, overwrite_version=self.version)
        if save_image == True or save_video == True:
            os.makedirs("evaluations", exist_ok=True)
            self.logger.text_name = evaluation_name + ".txt"
            self.logger.logging_directory = "evaluations"
            self.logger.mode = tud_presence_prediction.helpers.logging.logger_mode.TEXT
        
        # begin evaluation preparation
        self.logger.headline(f"Starting evaluation preperation.")
        self.logger.info(f"Overwriting default prediction settings with {str(prediction_settings)}.")

        test_datasets = []
        user_ids = []
        original_set_count = 0
        total_length = 0

        # accumulate test data and send it to the correct device
        for user_index in range(len(self.datasets)):
            for dataset_type in range(len(self.datasets[user_index])):
                if self.datasets[user_index][dataset_type] == None: continue
                if dataset_type == 0 and "training" not in evaluation_split: continue
                if dataset_type == 1 and "validation" not in evaluation_split: continue
                self.datasets[user_index][dataset_type].tensors = tuple(subtensor.to(self.used_accelerator) for subtensor in self.datasets[user_index][dataset_type].tensors)

                ds_len = len(self.datasets[user_index][dataset_type])
                if ds_len != 0: 
                    test_datasets.append(self.datasets[user_index][dataset_type])
                    user_ids.append(user_index)
                    original_set_count += 1
                    total_length += ds_len

        assert len(test_datasets) > 0, "No datasets of the requested catergory have been procured."

        # remove users who's datasets are too short as they restrict the available space for offset shifts and as larger sets will have too much of an impact on the results
        if user_treshold != 0 and user_treshold != None:
            average_set_length = total_length/original_set_count
            self.logger.info(f"Average set length before threshold filtering: {average_set_length}.")
            length_threshold = user_treshold * average_set_length
            self.logger.info(f"Removing sets that have a length of less than {length_threshold}.")
            previous_user_count = len(test_datasets)
            fitting_user_collection = []
            new_user_ids = []
            for set_position, user_set in enumerate(test_datasets):
                if len(user_set) >= length_threshold: 
                    fitting_user_collection.append(user_set)
                    new_user_ids.append(user_ids[set_position])
            test_datasets = fitting_user_collection
            user_ids = new_user_ids
            
            self.logger.info(f"Removed {previous_user_count - len(test_datasets)} datasets.")

        assert len(test_datasets) > 0, "No datasets matching the requested criteria have been procured."

        # get dataset lenghts to determine possible test cases later
        test_dataset_lengths = [len(dataset) for dataset in test_datasets]
        max_dataset_size = min(test_dataset_lengths) if equalize_user_representation == True else max(test_dataset_lengths) # either use smallest or largest dataset as the limiting factor
        self.logger.info(f"Limiting offset according to a maximum dataset size of {max_dataset_size} as defined by the {'largest' if equalize_user_representation == False else 'smallest'} set.")

        # determine if model supports predictions of variable length
        variable_prediction_length_support = model_util.get_model_variable_prediction_length_support(self.model)    
        assert variable_prediction_length_support == True or prediction_length_step == prediction_length_max, "Performing predictions of variable length is only supported for regressive models."
        assert variable_prediction_length_support == True or (self.data_time_shift != None and self.data_time_shift != (0,0)), "Model is non-regressive and no time-shift has been provided."
        label_offset = variable_prediction_length_support # used to determine if label data is contained in the input data, which is the case for time shift based models and data. Otherwise label data is taken from a slice after the input data's slice
        
        # get additiona information about the data and the model
        input_data_cut = model_util.get_model_data_cut(self.model)
        timeslots_per_day = test_datasets[0][0][0].size(0)

        # configure the models mask if it offers dynamic masking
        if force_mask != False:
            if model_util.get_model_mask_support(self.model):
                model_util.set_model_mask(self.model, force_mask)
                self.logger.info(f"Set models mask to {force_mask}.")
            else:
                self.logger.info(f"Model does not support dynamic masking.")

        # determine test case variation parameters
        prediction_length_max = prediction_length_max      
        prediction_length_step = prediction_length_step    
        prediction_length_variations = prediction_length_max/prediction_length_step
        assert prediction_length_variations == int(prediction_length_variations), "Maximum prediction length must be divisible by prediction length steps."
        assert show_user_presence_prediction == None or show_user_presence_prediction[1] <= prediction_length_max, "Can not generate user specific presence predictions for predictions lengths larger then the given maximum prediction length."
        prediction_length_variations = int(prediction_length_variations)

        window_size_max = self.sequence_size[0]
        window_size_step = window_size_step
        window_size_variations = window_size_max/window_size_step
        assert window_size_variations == int(window_size_variations), "Sequence length must be divisible by window size step size."
        assert input_data_cut == None or window_size_step > input_data_cut, "Step size for window size variations must be greater than maximum prediction lenght for regressive models."
        assert show_user_presence_prediction == None or show_user_presence_prediction[0] <= window_size_max, "Can not generate user specific presence predictions for window sizes larger then the given maximum window size."
        window_size_variations = int(window_size_variations)

        window_offset_max = (max_dataset_size - window_size_max) - (prediction_length_max if label_offset == True else 0) # could be set to max(...) to utilize as much data as possible, but larger datasets will have a stronger impact on the results and the offset evaluation becomes ambivalent as larger offsets are only possible for larger datasets
        window_offset_max = window_offset_max - (window_offset_max % window_offset_step)
        window_offset_step = window_offset_step
        window_offset_variations = (window_offset_max/window_offset_step) + 1
        assert window_offset_variations == int(window_offset_variations), "Sequence length must be divisible by window offset step size." # TODO: allow for every type of offset step/stride?
        assert window_offset_variations > 1, "No test case can be generated as the smallest dataset within the procured sets was too small. Increasing the user threshold parameter may resolve this issue."
        window_offset_variations = int(window_offset_variations)

        self.logger.headline(f"Evaluating {len(test_datasets)} datasets.")
        self.logger.info(f"Varying window size from {window_size_step} to {window_size_max} days{' (-' + str(input_data_cut) + ' adjustment)' if input_data_cut != None else ''}.")
        self.logger.info(f"Varying window offset from 0 to {window_offset_max} days.")
        self.logger.info(f"Varying prediction length from {prediction_length_step} to {prediction_length_max} days.")
        self.logger.info(f"Performing up to {window_size_variations} * {window_offset_variations} * {prediction_length_variations} = {window_size_variations*window_offset_variations*prediction_length_variations} prediction variations per user.")

        # deactivate logging for following predictions
        prev_logger_mode = self.logger.mode
        self.logger.mode = tud_presence_prediction.helpers.logging.logger_mode.NONE

        # prepare model
        self.model.to(self.used_accelerator)
        self.model.eval()
        with torch.no_grad():

            # generate arrays to hold all tracked results
            results = np.full((window_size_variations, window_offset_variations, prediction_length_variations), None)
            if show_user_presence_prediction != None: 
                full_user_labels = np.full((len(test_datasets), window_offset_max * timeslots_per_day + timeslots_per_day), np.nan, dtype=float)
                full_user_prediction = np.full((len(test_datasets), window_offset_max * timeslots_per_day + timeslots_per_day), np.nan, dtype=float)

            # iterate all test cases
            total_iteration_counter = 0
            skipped_iteration_counter = 0 
            for i_user in tqdm(range(len(test_datasets)), desc="User", position=0, unit_scale=False, bar_format="{desc:<20}{percentage:3.0f}%|{bar:90}|{n_fmt:>3s}/{total_fmt:3}", colour="#f7f5f2"):
                for i_window_size_index in tqdm(range(0, window_size_variations), desc="Window Size", position=1, leave=False, unit_scale=(window_size_step if window_size_step > 1 else False), bar_format="{desc:<20}{percentage:3.0f}%|{bar:90}|{n_fmt:>3s}/{total_fmt:3}", colour="#eae5de"):
                    i_window_size = window_size_step + (i_window_size_index * window_size_step)
                    for i_window_offset_index in tqdm(range(0, window_offset_variations), desc="Window Offset", position=2, leave=False, unit_scale=(window_offset_step if window_offset_step > 1 else False), bar_format="{desc:<20}{percentage:3.0f}%|{bar:90}|{n_fmt:>3s}/{total_fmt:3}", colour="#ddd6ca"):
                        i_window_offset = i_window_offset_index * window_offset_step
                        for i_prediction_length_index in tqdm(range(0, prediction_length_variations), desc="Prediction Length", position=3, leave=False, unit_scale=(prediction_length_step if prediction_length_step > 1 else False), bar_format="{desc:<20}{percentage:3.0f}%|{bar:90}|{n_fmt:>3s}/{total_fmt:3}", colour="#d0c7b6"):
                            i_prediction_length = prediction_length_step + (prediction_length_step * i_prediction_length_index)

                            # determine input data slice
                            i_input_window_start = i_window_offset
                            i_input_window_end = i_window_offset + i_window_size

                            # determine label data slice; done differently for time-shift based models as their labels are contained in the same window as the input data
                            i_label_window_start = (i_input_window_end - i_prediction_length) if label_offset == False else i_input_window_end
                            i_label_window_end = i_input_window_end if label_offset == False else (i_input_window_end + i_prediction_length)

                            
                            if i_label_window_end > len(test_datasets[i_user]): 
                                skipped_iteration_counter += 1
                                continue # end of users data reached

                            # get appropriate input data slice for this test iteration without labels
                            i_input_data = test_datasets[i_user][i_input_window_start:i_input_window_end]
                            i_input_data = (*(i_input_data[0:-1]), None)

                            # get appropriate label data slice for this test iteration 
                            i_label_data = test_datasets[i_user][i_label_window_start:i_label_window_end]
                            i_label_data = i_label_data[-1]

                            # perform prediction using real prediction settings
                            i_raw_prediction, i_prediction, reliability = self.predict(prediction_length_overwrite=i_prediction_length, processed_input_data_overwrite=i_input_data, log_results=False, **prediction_settings)

                            # regressive models roll their output to conform to the productive prediction standardization (prediction always begins at the last available day), so we need to roll it back to get the full prediction data
                            if variable_prediction_length_support == True: 
                                i_prediction = i_prediction.roll(-1, dims=0)

                            # sanity check prediction results
                            if None in reliability or "None" in reliability or "Z" in reliability:
                                raise RuntimeError("Unexpected prediction results.")
                            if i_input_data[0].size(0) != i_window_size or i_input_data[1].size(0) != i_window_size:
                                raise RuntimeError("Unexpected input data slice.")
                            if i_prediction.size(0) != i_prediction_length:
                                raise RuntimeError("Unexpected prediction lenghts.")
                            if i_prediction.shape != i_label_data.shape:
                                raise RuntimeError("Prediction shape does not match label shape.")

                            # only evaluate last prediction step 
                            if variable_prediction_length_support and prediction_length_step != prediction_length_max:
                                i_label_data = i_label_data[-prediction_length_step:]
                                i_prediction = i_prediction[-prediction_length_step:]

                            # prepare data for storage
                            i_prediction = i_prediction.clone().detach().flatten().cpu().numpy()
                            i_labels = i_label_data.clone().detach().flatten().cpu().numpy() 

                            label_mask = np.isnan(i_labels) # make sure only data with available labels is evaluated

                            if results[i_window_size_index][i_window_offset_index][i_prediction_length_index] == None:
                                results[i_window_size_index][i_window_offset_index][i_prediction_length_index] = {"prediction": [], "label": []}

                            if i_prediction[~label_mask].shape != i_labels[~label_mask].shape: raise TypeError("Prediction shape does not fit label shape.")

                            # store results for final evaluation
                            results[i_window_size_index][i_window_offset_index][i_prediction_length_index]["prediction"].extend(i_prediction[~label_mask].tolist())
                            results[i_window_size_index][i_window_offset_index][i_prediction_length_index]["label"].extend(i_labels[~label_mask].tolist())
                            
                            # potentially store results for user behaviour inspection
                            if show_user_presence_prediction != None and i_window_size == show_user_presence_prediction[0] and i_prediction_length == show_user_presence_prediction[1]:
                                full_user_labels[i_user][i_window_offset_index*timeslots_per_day:i_window_offset_index*timeslots_per_day + timeslots_per_day] = i_labels
                                full_user_prediction[i_user][i_window_offset_index*timeslots_per_day:i_window_offset_index*timeslots_per_day + timeslots_per_day] = i_prediction
                            
                            total_iteration_counter += 1

            self.logger.mode = prev_logger_mode
            self.logger.info(f"Skipped {skipped_iteration_counter} iterations.")
            self.logger.info(f"Performed {total_iteration_counter} iterations.")

            # show individual user presences before main evaluation plots if desired
            if show_user_presence_prediction != None:
                if save_image == True:
                    os.makedirs("evaluations", exist_ok=True)
                    os.makedirs(os.path.join("evaluations", "individual"), exist_ok=True)
                for usr in range(len(full_user_prediction)):
                    self.logger.generate_user_presence_graph(full_user_labels[usr], full_user_prediction[usr], timeslots_per_day=timeslots_per_day, title=f"Presences for User {user_ids[usr]}")
                    if save_image:
                        img_file_type = "png"
                        full_image_path = full_video_path = os.path.join("evaluations", "individual", self.logger.get_experiment_string(file_friendly=True, overwrite_version=self.version) + "_" + str(usr) + "." + img_file_type)
                        plt.savefig(full_image_path, bbox_inches='tight')
                
                plt.show(block=False)
                
            # Create figure with reserved slots for all desired plots
            graph_type_count = len(graph_types)
            fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), nrows=2, ncols=graph_type_count)
            
            # Set up additional visual elements
            fig.set_size_inches(20, 10)
            fig.suptitle(self.logger.get_experiment_string(overwrite_version=self.version), color="white")
            panel_alpha = 0.6
            if use_background == True: 
                fig.patches.extend([plt.Rectangle((0, 0), 0.999, 0.999, fill=True, facecolor=(0.094, 0.098, 0.101), hatch="ooo", alpha=1, zorder=-2, transform=fig.transFigure, figure=fig)])
            else:
                panel_alpha = 1
            fig.patches.extend([plt.Rectangle((0.125,0.51),0.78,0.43,
                                  fill=True, facecolor='#18191a', alpha=panel_alpha, zorder=-1, edgecolor="#000000", linewidth=2,
                                  transform=fig.transFigure, figure=fig)])
            fig.patches.extend([plt.Rectangle((0.125,0.05),0.78,0.43,
                                  fill=True, facecolor='#18191a', alpha=panel_alpha, zorder=-1, edgecolor="#000000", linewidth=2,
                                  transform=fig.transFigure, figure=fig)])
            
            header_color = (0.094, 0.098, 0.101, 1.0) if use_background == False else "white"
            first_panel_label = "Accuracy" if generate_recall == False else "Accuracy/\nRecall"
            fig.text(0.095, 0.725, first_panel_label, rotation="vertical", fontsize="xx-large", fontweight="bold", color=header_color, horizontalalignment="center", verticalalignment="center")
            fig.text(0.095, 0.265, "Precision", rotation="vertical", fontsize="xx-large", fontweight="bold", color=header_color, horizontalalignment="center", verticalalignment="center")

            # set vertical position of all graphs to generate properly positioned rows
            for pos in range (0, graph_type_count):
                full_pos = (1, pos) if graph_type_count > 1 else 1
                l, b, w, h = ax[full_pos].get_position().bounds
                ax[full_pos].set_position([l, 0.09, w, h])
            for pos in range (0, graph_type_count):
                full_pos = (0, pos) if graph_type_count > 1 else 0
                l, b, w, h = ax[full_pos].get_position().bounds
                ax[full_pos].set_position([l, 0.55, w, h])

            # function to get unweighted average score across one dimension/one slice of the original array to reduce dimensionality
            def get_average_scores(one_d_array, score_type):
                sum_score = 0
                valid_days = 0
                for row in one_d_array:
                    cur_score = 0
                    if score_type == "Precision": 
                        cur_score = precision_score(row["label"], row["prediction"], zero_division=np.nan)   
                    elif score_type == "Accuracy":
                        cur_score = accuracy_score(row["label"], row["prediction"])
                    elif score_type == "Recall":
                        cur_score = recall_score(row["label"], row["prediction"], zero_division=np.nan)
                    else:
                        raise ValueError("Invalid metric type.")

                    # catch case where precision can't be calculated when there are no positive labels
                    if not np.isnan(cur_score): 
                        valid_days += 1
                        sum_score += cur_score

                if valid_days == 0: return np.nan
                else: return (sum_score/valid_days)

            # function used to generate all graphs
            def generate_graph_6D_to_3D(figure, subplot, data_6D, flatten_axis, x_label, y_label, x_step, y_step, x_zero_based, y_zero_based, output_dim, verbalize=False, graph_color=None, edgecolor=None):

                # flatten and calculate score across one dimension; this reduces dimensionality by 2 since scores are calculated across all timeslots and then averaged across one dimension
                results3D = np.apply_along_axis(get_average_scores, flatten_axis, data_6D, output_dim)

                original_y_axis_variations = results3D.shape[0]
                original_x_axis_variations = results3D.shape[1]

                # matplotlib doesn't like plotting an axis of length 1 as that reduces the 3D plot to a 2D plot, so duplicate that dimension to have two identical entries. Only neccessary when one value is kept constant due to evaluation settings.
                Z = results3D
                if 1 == Z.shape[0]: Z = np.concatenate((Z, Z), axis=0)
                if 1 == Z.shape[1]: Z = np.concatenate((Z, Z), axis=1)

                y_axis_variations = Z.shape[0]
                x_axis_variations = Z.shape[1]

                # Creating meshgrid; required by matplotlib to draw surfaces in 3D space
                x = np.linspace(0, x_axis_variations -1, x_axis_variations)
                y = np.linspace(0, y_axis_variations -1, y_axis_variations) # Check if correct contents
                X, Y = np.meshgrid(x, y)

                # generate one surface plot
                if use_colormap == False: 
                    ls = LightSource(90, 10)
                    subplot.plot_surface(X, Y, Z, rstride=1, cstride=1, lightsource=ls, shade=True, antialiased=True, color="white", alpha=0.85, edgecolor=(1,1,1,1)) #antialiased=False, shade=False) rstride=0.1, cstride=0.1, 
                else: 
                    if edgecolor == None: edgecolor = (1,1,1,1)
                    if graph_color == None: graph_color = cm.Greys_r
                    else: 
                        color_steps = 100
                        vals = np.ones((color_steps, 4))
                        vals[:, 0] = np.linspace(graph_color[0], 0, color_steps)
                        vals[:, 1] = np.linspace(graph_color[1], 0, color_steps)
                        vals[:, 2] = np.linspace(graph_color[2], 0, color_steps)
                        graph_color = ListedColormap(vals)
                    
                    subplot.plot_surface(X, Y, Z, rstride=1, cstride=1, shade=True, antialiased=True, cmap=graph_color, norm=Normalize(vmin=0.2, vmax=1), alpha=0.85, edgecolor=edgecolor)

                # remember for animation    
                subplot.custom_x = X
                subplot.custom_y = Y
                subplot.custom_z = Z

                # calculate appropriate axis labels
                x_label_step = math.ceil(original_x_axis_variations/9)
                y_label_step = math.ceil(original_y_axis_variations/9)

                # configure visual elements for this plot
                self.logger._set_plot_style(subplot, figure, bright_text=True)
                output_dim_name = output_dim
                if output_dim == "Recall": output_dim_name = output_dim + " / Accuracy"
                subplot.set_title(f"{x_label}, {y_label} → {output_dim_name}", color="white")
                subplot.patch.set_alpha(0.0)
                subplot.xaxis.set_ticks(range(0, original_x_axis_variations, x_label_step))
                subplot.yaxis.set_ticks(range(0, original_y_axis_variations, y_label_step))
                subplot.xaxis.set_ticklabels([(i * x_step + (x_step if x_zero_based==False else 0)) for i in range (0, original_x_axis_variations, x_label_step)])
                subplot.yaxis.set_ticklabels([(i * y_step + (y_step if y_zero_based==False else 0)) for i in range (0, original_y_axis_variations, y_label_step)])
                subplot.xaxis.set_label_text(f"{x_label} in days")
                subplot.yaxis.set_label_text(f"{y_label} in days")
                subplot.set_zlabel(f"{output_dim_name}")
                subplot.set_zlim(0, 1)
                if draw_horizontal_contours: 
                    subplot.set_xlim(-1, x_axis_variations - 1)
                    subplot.set_ylim(0, y_axis_variations)
                    subplot.contour(X, Y, Z, zdir='y', offset=x_axis_variations + 2, levels=1, colors="#ddd6ca")
                    subplot.contour(X, Y, Z, zdir='x', offset=-1, levels=1, colors="#ddd6ca")
                if draw_vertical_contours:
                    subplot.contour(X, Y, Z, zdir='z', offset=0, cmap='gray')

                # generate additional text outputs if desired
                if verbalize:
                    self.logger.headline(f"{output_dim} results for {x_label}, {y_label}:")
                    for ax_one in range(results3D.shape[0]):
                        for ax_two in range(results3D.shape[1]):
                            input_varY = y_step * ax_one + (y_step if y_zero_based == False else 0)
                            input_varX = x_step * ax_two + (x_step if x_zero_based == False else 0)
                            self.logger.info(f"{y_label} {input_varY}, {x_label} {input_varX}: {results3D[ax_one][ax_two]*100}%")

            # get proper indexing for differently generated matplotlib ax arrays
            accuracy_ax_indices = [(0, acc_i) for acc_i in range (0,3)] if graph_type_count > 1 else (0,)
            precision_ax_indices = [(1, prec_i) for prec_i in range (0,3)] if graph_type_count > 1 else (1,)

            # generate accuracy plots
            if 1 in graph_types: generate_graph_6D_to_3D(fig, ax[accuracy_ax_indices[0]], results, 1, "Prediction Length", "Input Length", prediction_length_step, window_size_step, False, False, "Accuracy", verbalize=verbalize_results)
            if 2 in graph_types: generate_graph_6D_to_3D(fig, ax[accuracy_ax_indices[1]], results, 0, "Prediction Length", "Input Offset", prediction_length_step, window_offset_step, False, True, "Accuracy", verbalize=False)
            if 3 in graph_types: generate_graph_6D_to_3D(fig, ax[accuracy_ax_indices[2]], results, 2, "Input Offset", "Input Length", window_offset_step, window_size_step, True, False, "Accuracy", verbalize=False)
            # generate precision plots
            if 1 in graph_types: generate_graph_6D_to_3D(fig, ax[precision_ax_indices[0]], results, 1, "Prediction Length", "Input Length", prediction_length_step, window_size_step, False, False, "Precision", verbalize=verbalize_results)
            if 2 in graph_types: generate_graph_6D_to_3D(fig, ax[precision_ax_indices[1]], results, 0, "Prediction Length", "Input Offset", prediction_length_step, window_offset_step, False, True, "Precision", verbalize=False)
            if 3 in graph_types: generate_graph_6D_to_3D(fig, ax[precision_ax_indices[2]], results, 2, "Input Offset", "Input Length", window_offset_step, window_size_step, True, False, "Precision", verbalize=False)
            # generate recall plots
            if generate_recall == True:
                recall_graph_color = (0.815, 0.780, 0.713) # (0.917, 0.898, 0.870)
                recall_edge_color = (0.815, 0.780, 0.713)
                if 1 in graph_types: generate_graph_6D_to_3D(fig, ax[accuracy_ax_indices[0]], results, 1, "Prediction Length", "Input Length", prediction_length_step, window_size_step, False, False, "Recall", verbalize=verbalize_results, graph_color=recall_graph_color, edgecolor=recall_edge_color)
                if 2 in graph_types: generate_graph_6D_to_3D(fig, ax[accuracy_ax_indices[1]], results, 0, "Prediction Length", "Input Offset", prediction_length_step, window_offset_step, False, True, "Recall", verbalize=False, graph_color=recall_graph_color, edgecolor=recall_edge_color)
                if 3 in graph_types: generate_graph_6D_to_3D(fig, ax[accuracy_ax_indices[2]], results, 2, "Input Offset", "Input Length", window_offset_step, window_size_step, True, False, "Recall", verbalize=False, graph_color=recall_graph_color, edgecolor=recall_edge_color)

            if show_live: 
                #manager = plt.get_current_fig_manager()
                #manager.full_screen_toggle()
                plt.show()
                

            if save_image == True or save_video == True:
                if save_image:
                    img_file_type = "png"
                    full_image_path = full_video_path = os.path.join("evaluations", evaluation_name + "." + img_file_type)
                    fig.savefig(full_image_path, bbox_inches='tight')
                    self.logger.info(f"Image stored to {full_image_path}.")
                if save_video:
                    def update(frame):
                        angle_norm = (frame + 180) % 360 - 180
                        for row_index in range(0,2):
                            if graph_type_count > 1:
                                for column_index in range(0, graph_type_count):
                                    ax[row_index, column_index].view_init(ax[row_index, column_index].elev, angle_norm)#, roll+1)
                            else:
                                ax[row_index].view_init(ax[row_index].elev, angle_norm)#, roll+1)

                        rotate_light = False # inconsistent results
                        if rotate_light:
                            #new_surf_color = new_ls.shade_rgb(np.ones((ax[0,1].custom_z.shape[0], ax[0,1].custom_z.shape[1], 3)), ax[0,1].custom_z, blend_mode="soft", vert_exag=2) # can be used instead of lightsourced itself with facecolors=new_surf_color
                            for subplot in np.nditer(ax): 
                                if hasattr(subplot, "custom_surface"): subplot.custom_surface.remove()
                                new_ls = LightSource(90 - frame, 0)
                                subplot.custom_surface = subplot.plot_surface(subplot.custom_x, subplot.custom_y, subplot.custom_z, rstride=1, cstride=1, lightsource=new_ls, shade=True, antialiased=True, color="white", alpha=0.85, edgecolor=(1,1,1,1)) 

                    self.logger.info("Generating animation file. This may take several minutes.")

                    video_file_type = "gif"
                    full_video_path = os.path.join("evaluations", self.logger.get_experiment_string(file_friendly=True, overwrite_version=self.version) + "." + video_file_type)

                    ani = animation.FuncAnimation(fig=fig, func=update, frames=360, interval=50)
                    ani.save(filename=full_video_path, writer="pillow")
                    self.logger.info(f"Video stored to {full_video_path}.")
                

    def predict(self, log_results=True, generate_reliability_level=True, post_processing="binary percentage cutoff", post_processing_threshold=0.6, adjust_data_length=True, prioritize_padding=False, processed_input_data_overwrite=None, prediction_length_overwrite=None):
        """ 
        Performs a prediction based on the initially given parameters. Additional parameters determine logging, generation of processed results and pre-processing of input data that may improve prediction quality.
        The returned data always refers to all days between the current day (inclusive) and all requested days. E.g. if a prediction of length 2 is performed, all timeslots of the current day and the next day are returned.
        
        Args:
            log_results (bool): 
                If set to True, a console log will reflect data and time, the original prediction value, the processed prediction value, verbalization and reliability levels.
            generate_reliability_level (bool):
                If set to True, an additional numpy array is returned which contains information about the reliability of every time slot contained in the prediction, structured in the same way. It is highly recommended to make use of this information, for test cases and for productive predictions alike. 
                Reliability is given as alphabetical characters, in descending order. 
                For models with variable prediction lenghts, this begins at 'A' and changes to the next character after every internal prediction step. Subsequent predictions should slightly decrease in quality.
                For time-shift based models, this begins at 'A' and changes to 'Z' immediately when the window the model is trained on is exhausted. For these models, only the predictions within the trained time frame are expected to be reliable. 
                For any time-slots before the current point in time, the reliability-level is "None". These do not represent predictions and they should never be used.
            post_processing (str):
                If this is not None, then the second return parameter will be an additional tensor containing a processed version of the prediction.
                Supported values:
                    "binary cutoff": Converts the predictions into a binary format neccessary for final evaluation by simply using the post processing threshold as a cutoff point at which results are put into categories 0 or 1. With the threshold set to 0.5, this represents the default case.
                    "binary daily percentile": Converts the predictions into a binary format by only assuming presence for the top values in each day. The threshold represents the percentage of timeslots in each day which should be considered present. An additional static check is performed to ensure that all of these are above 0.5.
                    "binary percentage cutoff": Converts the predictions into percentages by applying a sigmoid function. These percentages are then converted to binary by using the threshold value as a cutoff.
                    "continous percentages": Converts the predictions into percentages by applying a sigmoid function. As the result is still a decimal number, these results do not directly indicate presence or absence.
            post_processing_threshold (float):
                An additional parameter used for post-processing. Refer to the individual post-processing methods for more information. 
            adjust_data_length (bool):
                If set to True, the function will try to intelligently figure out the best input data format based on the amount of data and the initially given sequence lengths. This may affect prediction quality.
            prioritize_padding (bool):
                If set to True, the function will prioritize larger input size categories when automatically adjusting data length. This may affect prediction quality. Has no effect if automatic adjustment is turned off. 
            processed_input_data_overwrite (TensorDataset):
                Used evaluation-calls only; not intended for productive use.
            prediction_length_overwrite (int):
                Used evaluation-calls only; not intended for productive use.
                
        Returns:
            (Tensor, Tensor, NumpyArray): Raw prediction, processed prediction (optional), reliability (optional)
            Each of these collections contain all prediction information in shape [days, timeslots]. The first entry in the collection refers to the day the prediction was performed on. Following entries refer to following days.
        """
        
        self.logger.headline("Starting Prediction")

        # sanity check version and extrapolation
        if self.extrapolation == "none" or self.extrapolation == None:
            self.logger.critical("Warning: Using no data extrapolation for predictions. This may lead to unexpected results for users whose data has not been updated recently. Productive predictions should always be performed with extrapolation enabled.") 
        if self.version == None:
            raise ValueError("No version has been provided and no trained model could be loaded.")

        # get start date
        start_date = date.fromisoformat(str(self.date)) if self.date != None else (date.today() if self.extrapolation == "now" else None)
        self.logger.info(f"Prediction starting from {start_date}.")
        if start_date == None and processed_input_data_overwrite == None:
            raise ValueError("No prediction date has been provided.")
    
        # get desired prediction length
        prediction_length = self.days if prediction_length_overwrite == None else prediction_length_overwrite # overwrite is used for evaluation calls only
        self.logger.info(f"Predicting {prediction_length} days.")
        if prediction_length == None:
            raise ValueError("No prediction length has been provided.")

        # sanity check prediction length settings and configure correct prediction length
        variable_prediction_length_support = model_util.get_model_variable_prediction_length_support(self.model)
        data_shift_days = None
        if variable_prediction_length_support == False:
            # shift-based models have implicit prediction length settings determined by data procurement (data is shifted and extra days are appended). No additional configuration is required at this point.
            data_shift_days = data_processing_util.interpret_shift_string(self.data_time_shift)[0]
            if data_shift_days == None:
                raise ValueError("Model does not support variable prediction lengths and no shift string has been provided.")
            if prediction_length != data_shift_days:
                self.logger.critical("Warning: Desired prediction length does not match time shift.")
        elif variable_prediction_length_support == True:
            # models without data shift must have their prediction length property configured explicitly.
            model_util.set_model_prediction_length(self.model, prediction_length)
            if data_shift_days != None:
                self.logger.critical("Warning: Performing prediction of variable length with a time shift.") # this may be a viable scenario in the future if models are implemented that support variable prediction length while still using time-shifts. Currently all models only use either one.

        # get full user data
        user_data_set = self.datasets[0][0]
        full_user_data = user_data_set[:][0:-1] if processed_input_data_overwrite == None else processed_input_data_overwrite[0:-1] # overwrite is used for evaluation calls only
        max_user_data_length = len(full_user_data[0])
        self.logger.info(f"Original user data size: {max_user_data_length} days.")

        # check amount of necessary extrapolation as higher amounts indicate outdated user data. Prediction quality may decline rapidly when data is outdated as the model learns to pay signficiant amounts of attention to recent coordinates as they are always available during training.
        dataset_metadata = data_processing_util.get_dataset_metadata(user_data_set)
        if dataset_metadata["extrapolated_slots"] != None: 
            self.logger.info(f"Extrapolated {int(dataset_metadata['extrapolated_slots'])} time slots.")

        # get appropriate user data slice; data may be cut to fit into smaller categories or padded to fit into larger categories for sequence sizes the model has explicitly been trained on in order to improve results
        self.sequence_size.sort(reverse=True)
        desired_input_length = self.sequence_size[0]
        used_data_cut = 0
        if adjust_data_length:
            model_data_cut = model_util.get_model_data_cut(self.model) # if model slices the data during the training process, we may want to reproduce training cirucumstances to improve prediction results
            used_data_cut = model_data_cut if (model_data_cut != None and adjust_data_length == True) else 0
            possible_sequence_sizes = [trained_seq_len - used_data_cut for trained_seq_len in self.sequence_size]
            largest_containing_size_index = len(possible_sequence_sizes) -1 # start with smallest possible size
            for trained_sequence_size_index, trained_sequence_size in enumerate(possible_sequence_sizes):
                if max_user_data_length >= trained_sequence_size and trained_sequence_size > possible_sequence_sizes[largest_containing_size_index]:
                    largest_containing_size_index = trained_sequence_size_index
            
            # if padding is prioritized for predictions, force user into larger category if he's inbetween categories
            if prioritize_padding == True and (largest_containing_size_index - 1) >= 0 and max_user_data_length > possible_sequence_sizes[largest_containing_size_index]: desired_input_length = possible_sequence_sizes[largest_containing_size_index - 1]
            # always use largest category that is still able to hold the user without padding
            else: desired_input_length = possible_sequence_sizes[largest_containing_size_index]

        # cut user data to maximum length
        input_data = tuple([full_user_data_tensor[-desired_input_length:] for full_user_data_tensor in full_user_data])
        model_util.set_model_desired_input_length(self.model, desired_input_length)
        self.logger.info(f"Using {len(input_data[0])} days of user input data (category {desired_input_length} with {used_data_cut * -1} adjustment).")
        
        # potentially add padding
        if adjust_data_length and len(input_data[0]) < desired_input_length:
            required_padding = desired_input_length - len(input_data[0])
            input_data = data_processing_util.add_padding(input_data, required_padding, label_tensor_index=None)
            self.logger.info(f"Added {required_padding} padding for a total length of {len(input_data[0])}.")

        # move model and data to desired accelerator
        if str(self.model.device) != self.used_accelerator: self.model.to(self.used_accelerator)
        if str(input_data[0].device) != self.used_accelerator: input_data = tuple(input_tensor.to(self.used_accelerator) for input_tensor in input_data)

        # perform prediction
        prediction = None
        self.model.eval()
        with torch.no_grad():    
            prediction = self.model(*input_data) # input_data is a tuple of tensors (location, calendar) where location is a tensor of shape (batch_size, sequence_length, location_dim) and calendar is a tensor of shape (batch_size, sequence_length, calendar_dim)
        self.logger.info(f"Generated prediction of shape {prediction.shape}.")

        # adjust output to consistent format (different models may yield different formats and timespans)
        prediction = model_util.standardize_output(prediction, input_data[0].size(1))
        self.logger.info(f"Standarized output to shape {prediction.shape}.")

        # sanity check prediction results
        if prediction.size(0) != input_data[0].size(0) and prediction.size(0) != prediction_length:
            raise RuntimeError("Unexpected prediction results.")

        # cut results 
        prediction = prediction[-prediction_length:]

        # perform post-processing on results if desired
        processed_prediction = prediction
        if post_processing != None:
            if str.lower(post_processing) != "continous percentages" and (type(post_processing_threshold) != float and type(post_processing_threshold) != int):
                # all post-processing methods that include a binary conversion require information about the threshold
                raise ValueError("Invalid post processing limit.") 
            
            if str.lower(post_processing) == "binary cutoff":
                processed_prediction = (prediction > post_processing_threshold).float()
                processed_prediction[prediction.isnan()] = float('nan')
            elif str.lower(post_processing) == "binary daily percentile":
                processed_prediction = torch.zeros_like(prediction)
                desired_result_count = prediction.size(1) * post_processing_threshold
                top_results, top_result_indices = torch.topk(prediction, int(desired_result_count), dim=-1) # get daily top values
                for day_index, day in enumerate(processed_prediction):
                    daily_top_indices = top_result_indices[day_index]
                    daily_top_indices = daily_top_indices[top_results[day_index] > 0.5] # additional static filter to prevent results that are still too low
                    day[daily_top_indices] = 1
                processed_prediction[prediction.isnan()] = float('nan')
            elif str.lower(post_processing) == "binary percentage cutoff":
                processed_prediction = torch.sigmoid(prediction)
                processed_prediction = (processed_prediction > post_processing_threshold).float()
                #processed_prediction[prediction.isnan()] = float('nan')
            elif str.lower(post_processing) == "continous percentages":
                processed_prediction = torch.sigmoid(prediction)
            else:
                # if more post processing variety is desired, add more methods here
                raise ValueError("Invalid post processing method given.")
                
        # generate reliability level if desired
        reliability = None
        if generate_reliability_level == True:
            prediction_start_slot = None
            if self.extrapolation == None or self.extrapolation == "none":
                prediction_start_slot = 0
                self.logger.critical("Trying to generate reliability levels when no extrapolation is used is unreliable. Assuming prediction of full days.")
            else: 
                # days that are not contained in the procured input data, e.g. sundays
                if datetime.now().date().weekday() not in dataset_metadata["weekdays"]: 
                    prediction_start_slot = prediction.size(1)          
                # all other days and daytimes
                else: 
                    daily_start_time = time.fromisoformat(str(dataset_metadata["daily_min_time"]))
                    daily_start_time = timedelta(hours=daily_start_time.hour, minutes=daily_start_time.minute)
                    daily_end_time = time.fromisoformat(str(dataset_metadata["daily_max_time"]))
                    daily_end_time = timedelta(hours=daily_end_time.hour, minutes=daily_end_time.minute)
                    current_time = np.clip(timedelta(hours=datetime.now().time().hour, minutes=datetime.now().time().minute).total_seconds(), daily_start_time.total_seconds(), daily_end_time.total_seconds()) - daily_start_time.total_seconds()
                    max_time = (daily_end_time - daily_start_time).total_seconds()

                    daytime_percentage = current_time / max_time
                    prediction_start_slot_sanity_check = math.floor((prediction.size(1) - 1) * daytime_percentage) + 1 # TODO: Better concept to standardize results across models. This will yield different results for prediction between 20:45-23:59 and 00:01-8:00, but other check will not
                    #prediction_start_slot = (prediction.size(1) - dataset_metadata["missing_slots"]) % prediction.size(1) 
                    prediction_start_slot = prediction_start_slot_sanity_check
                    slot_delta = abs(prediction_start_slot - prediction_start_slot_sanity_check)
                    
                    if slot_delta > 1:
                        raise ValueError("Correct starting time for the prediction could not be determined.")
                    elif slot_delta == 1:
                        self.logger.critical("Correct starting time for prediction is ambiguous. This should only occur when the prediction has been performed between two time slots.")

            reliability_levels = [str(char).upper() for char in list(map(chr, range(97, 123)))] # alphabet
            reliability = np.empty(math.prod(prediction.shape), dtype="U4") # create a second tensor as an equivalent of the flattened prediction tensor
            reliability[:prediction_start_slot] = None # demarkate non-prediction slots
            # time shift based models should exhibit a steep decline in reliability beyond the size of the time shift they have been trained for as they will not have any coordinates available for these time slots.
            if variable_prediction_length_support == False and data_shift_days != None and data_shift_days > 0:
                cutoff_slot = prediction_start_slot + prediction.size(1) * data_shift_days
                reliability[prediction_start_slot:cutoff_slot] = reliability_levels[0]
                if cutoff_slot < reliability.size: 
                    reliability[cutoff_slot:] = reliability_levels[-1]  # Only discount the reliability level once, immediatly going to the last level.
            # regressive models have a continous decline in prediction quality based on their internal regressive predicition steps
            else:
                prediction_step = model_util.get_model_prediction_step_length(self.model)
                print(f"DEBUG: prediction_step = {prediction_step}, Typ: {type(prediction_step)}")
                print(f"DEBUG: prediction size = {prediction.size()}, Typ: {type(prediction)}")
                prediction_step = prediction_step[0] * prediction.size(1) + prediction_step[1]
                last_slot = prediction_start_slot
                prediction_step_index = 0
                while last_slot < reliability.size:
                    next_slot = last_slot + prediction_step
                    if next_slot >= reliability.size: next_slot = reliability.size # don't go beyond the end of the array
                    last_slot = next_slot
                    prediction_step_index += 1
            reliability = reliability.reshape(prediction.shape) # mimic prediction shape

        # log results if desired
        if log_results: data_processing_util.log_presence(processed_prediction, start_date, self.logger, raw_predictions=prediction, reliability_levels=reliability)

        return prediction, processed_prediction, reliability


def main():
    # --- define command line arguments ---
    parser = argparse.ArgumentParser()

    # define task
    parser.add_argument("--train", action="store_true", help="When this flag is provided a new model will be trained.")
    parser.add_argument("--predict", action="store_true", help="When this flag is provided a presence prediction will be performed on a pre-trained model.")
    parser.add_argument("--evaluate", action="store_true", help="When this flag is provided multiple presence predictions with varied inputs will be performed on a pre-trained model.")

    # define data and model (mandatory for all tasks; default values will be provided if parameters are omitted)
    parser.add_argument("--data", action="store", help="Name of the data procurement module (OR DATA LOADER?), located in the 'data' sub-directory. This file must implement a class with the functions load() which loads and returns unprocessed data and the function process() to load processed data and return a DataLoader to be used by PyTorch. The procurer may optionally accept parameters for the household and datetime to load specific data.")
    parser.add_argument("--model", action="store", help="Name of the (lightning) module defining the model to be used, located in the 'models' sub-directory. The module must be contained in the given file and the module's class name must have the same name as file name.")

    # additional parameters for prediction; not neccessary for training
    parser.add_argument("--days", action="store", type=int, help="Mandatory to perform a prediction, optional for training. Must contain the amount of days you want to predict, including the current day.")
    parser.add_argument("--extrapolation", action="store", help="For real-life predictions, this should always be set to 'now'. Alternatively supports 'none' and 'last', only used when intentionally performing test prediction for past points in time in conjunction with the date parameter.")
    parser.add_argument("--date", action="store", help="Optional, mainly used for individual test predictions. Not will artifically prolong or truncate available data before processing.")

    # additional optional parameters for training; not neccessary for prediction 
    parser.add_argument("--epochs", action="store", type=int, help="The number of epochs the training should run for.")
    parser.add_argument("--tostorage", action="store_true", help="If this argument is given data should be saved to a local storage using the given DataProcurer.")
    parser.add_argument("--fromstorage", action="store_true", help="If this argument is given data should be loaded from a local storage using the given DataProcurer.")
    parser.add_argument("--storereadable", action="store_true", help="Warning: This may generate a high amount of local files. If this argument is provided then raw data and processed data will be stored additionally in a readable format, independently from other parameters. This data can not be used as an input ")
    parser.add_argument("--storagename", action="store", help="The name of the text file used to store/retrieve data from/to locally. Must be located within the 'data/storage'-directory. The filename must have the .txt extension and the parameter must be given with no file extension.")
    parser.add_argument("--split", action="store", type=int, help="Optional for predictions and training. This can be used to manually set the amount of validation data at the end of the procured samples, which will not be used for training. Given as a number between 0 and 100. For productive usage, set this to 0.")
    parser.add_argument("--accelerator", action="store", help="Used accelerator. Default value is auto, which will use CUDA cores if available, otherwise CPU. Options: 'cpu', 'gpu', 'tpu', 'ipu', 'auto'.")
    parser.add_argument("--batchsize", action="store", type=int, help="Determines how many sequences should be processed in parallel when training. This has a slight effect on training results, but is mainly useful to fully utilize available computing power by increasing batch size when additional ressources are available. Should be given in powers of two.")
    parser.add_argument("--stridesize", action="store", type=int, help="Determines how sequences are generated for training. Indicates how many days are between two subsequent input sequences of the same user. Lower values generate more variations of input sequences; this may improve generalization but will increase training time.")
    parser.add_argument("--cloud", action="store_true",  help="If this argument is given, the logging path will be adjusted for cloud training on GCP")

    # additional parameters for both training and predictions
    parser.add_argument("--version", action="store", help="Mandatory to perform a prediction, optional for training. Must contain the lightning version number of the directory containting the desired .ckpt file. This file will be used to reconstruct a trained model.")
    parser.add_argument("--shift", action="store", help="Optional for predictions and training. The amount of shift in unknown past data may be given in the format DAYS:TIMESLOTS. For example, if one wants to train a model to look one day into the future, use '--shift 1:0'.")
    parser.add_argument("--user", action="store", type=int, help="Internal user ID used to retrieve parameters for the webhook.")
    parser.add_argument("--sequencesize", action="store", type=int, nargs='+', help="Determines how many days of input make up one input sequence for one prediction. Multiple values may be given, separated by spaces; this will cause a signficant increase in sequence variations, usually by a factor larger than two, and consequently increases training time by the same factor. In turn, it might increase generalization performance on multiple sequences lengths. May be used to improve prediction performance on users with very small datasets. Should be given in powers of two.")

    # define logging behaviour to prevent spamming files on certain systems (e.g. if executed on server)
    parser.add_argument("--logging", action="store", choices=["full", "text", "console", "none"], help="Defines the amount of output generated during training or prediction. 'Console' will generate console outputs, 'text' will additionally save these to a file, 'full' will additionally generate and save images.")
    
    args_space = parser.parse_known_args()[0]
    processed_args = vars(args_space)

    print()
    print(f"--- Starting new presence prediction process ---")
    print(f"Arguments: {args_space}.")    
    print()

    # --- first sanity check for arguments  ---
    if sum((processed_args["train"], processed_args["predict"], processed_args["evaluate"])) > 1:
        print("Cannot train and predict at the same time. Only one of these arguments must be provided. Canceling execution.")
        return
    if sum((processed_args["train"], processed_args["predict"], processed_args["evaluate"])) < 1: 
        print("No action was specified. Defaulting to training mode.")
        processed_args["train"] = True
        processed_args["predict"] = False
        processed_args["evaluate"] = False
    if processed_args["predict"]:
        if not processed_args["version"]:
            print("Since a trained model is needed to perform a prediction, the version argument is mandatory but has been omitted. Canceling execution.")
            return
        if not processed_args["days"]:
            print("Since a prediction is performed for a specific length, the days argument is mandatory but has been omitted. Canceling execution.")
            return
        
    for filename in [processed_args["data"], processed_args["model"], processed_args["storagename"], processed_args["version"]]:
        if filename == None: continue

        cleanFileName = "".join(c for c in filename if (c.isalnum() or c == "_"))
        if filename != cleanFileName: 
            print("Invalid filename given. Canceling execution.")
            return
   
    # --- forward command line inputs to presence_prediction class instance  ---

    # create new presence prediction
    new_presence_prediction = presence_prediction(
        model_file = processed_args["model"], 
        data_procurer_file = processed_args["data"], 
        data_time_shift = processed_args["shift"],
        data_from_storage = processed_args["fromstorage"] if processed_args["fromstorage"] == True else None, 
        data_to_storage = processed_args["tostorage"] if processed_args["tostorage"] == True else None, 
        data_storage_filename = processed_args["storagename"], 
        data_to_readable_file = processed_args["storereadable"],
        version = processed_args["version"],
        epochs = processed_args["epochs"],
        logging_mode = processed_args["logging"],
        date = processed_args["date"],
        days = processed_args["days"],
        user = processed_args["user"],
        extrapolation = processed_args["extrapolation"], 
        split = processed_args["split"],
        batch_size = processed_args["batchsize"],
        sequence_size = processed_args["sequencesize"],
        stride_size = processed_args["stridesize"],
        accelerator = processed_args["accelerator"],
        cloud=processed_args["cloud"]
    )

    if processed_args["train"]: 
        # train the model
        trained_model = new_presence_prediction.train()
        return trained_model
    
    elif processed_args["predict"]:
        # perform a prediction 
        prediction, processed_prediction, reliability = new_presence_prediction.predict()
        return prediction, processed_prediction, reliability
    
    elif processed_args["evaluate"]:
        # perform an evaluation 
        new_presence_prediction.evaluate()
    


if __name__ == "__main__":
    main()
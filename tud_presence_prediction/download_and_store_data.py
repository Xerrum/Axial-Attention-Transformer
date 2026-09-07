from tud_presence_prediction.data.internal.data_processing import data_processing_util
from tud_presence_prediction.data.internal.data_procurer import data_procurer
import logging

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_and_store_data():
    # Set parameters for data loading
    data_procurer_file = "data_procurer"  # Replace with the actual data procurer file name (without .py)
    data_storage_filename = "auto"  # Use "auto" or specify a filename
    data_from_storage = False
    data_to_storage = True
    store_readable = True

    # Initialize data procurer
    data_procurer_instance = data_procurer(
        logger=logger,
        from_storage=data_from_storage,
        to_storage=data_to_storage,
        storage_file=data_storage_filename,
        store_readable=store_readable
    )

    # Load and store data (you can pass additional parameters if required)
    datasets = data_procurer_instance.load(
        shift=None,  # Add the appropriate shift if needed
        user=None,  # Specify user ID if necessary
        date=None,  # Provide a specific date or leave it as None
        days=None,  # Specify number of days if needed
        extrapolate="now",  # Or "last" or "none", based on your need
    )

    logger.info("Data has been downloaded and saved to storage.")

if __name__ == "__main__":
    download_and_store_data()

    # default settings for parameters
    default_model_file = "FlashAttention_V1"    # python file within the 'models' directory, containing a lightning module.
    default_data_procurer_file = "dynamic_multiuser"    # python file within the 'data' directory, containing a data_procurer.
    default_data_from_storage = False                   # wether data should be retrieved from local storage if available.
    default_data_to_storage = True                     # wether data should be saved to local storage.
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

# initialize data procurer
self.logger.headline(f"Loading data procurer '{self.data_procurer_file}'")
self.data_procurer_name = self.data_procurer_file
self.data_procurer_subclass = getattr(
    importlib.import_module(".data." + self.data_procurer_file, package="tud_presence_prediction"),
    self.data_procurer_name)  # TODO: do additional sanity checks (e.g. see if file is there and maybe in correct directory)
self.data_procurer = self.data_procurer_subclass(
    logger=self.logger,
    from_storage=self.data_from_storage,
    to_storage=self.data_to_storage,
    storage_file=self.data_storage_file,
    store_readable=self.data_to_readable_file
)
self.logger.info(f"Data procurer {self.data_procurer.__class__.__name__} loaded succesfully.")

# load data
self.datasets = self.data_procurer.load(self.data_time_shift, date=self.date, days=self.days, user=self.user,
                                        user_home_coordinates=self.user_home_coordinates, user_data=self.user_data,
                                        split=self.split, extra=self.data_time_shift, extrapolate=self.extrapolation)

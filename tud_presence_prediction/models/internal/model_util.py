import torch
import pytorch_lightning as pl

from sklearn.metrics import precision_score, accuracy_score
import numpy as np

class model_util:
    """
    Provides two types of model related functionality:
        - Functions used by by models internally, standardizing commonly needed features, such as tracking of desired metrics and configuring optimizers and learning rate schedulers
        - Functions used externally to gather information about models or to interact with models, such as setting their desired prediction horizon, setting the used type of masking or setting the used pos_weight values for loss calculation
    """

    batch_count_per_epoch = None    # must be set externally to determine LR curve. If this is not set but a LR scheduler is used that requires this, an error is raised.

    log_per_epoch = True            # wether logs are averaged per epoch. Lightning requires this to be True if checkpoints should represent the best result per Epoch.

    model_data_cut_hparam = "data_cut"                              # a model's Lightning hyper parameter that holds information about batch-preprocessing. Some models may cut batches into two parts while trying to predict the second one. Given in days.
    model_prediction_length_variable = "prediction_total_length"    # a model's attribute that determines the prediction length. If a model provides this attribute, it is assumed to support variable prediction lengths. Given in days.
    model_prediction_step_length_variable = "prediction_step_length" 
    model_mask_variable = "use_mask"                                # a model's attribute that determines which type of masking a model uses. Different models may implement different values for this variable.
    model_pos_weight_function = "adjust_pos_weights"                # a model's function to adjust pos weights.
    model_input_length_variable = "max_input_length"             # a models attribute that determines the sequence length that is desired to be used internally if the model alters sequence lengths

    @staticmethod
    def configure_logging(model: pl.LightningModule, log_lr=False, use_val_loop=None, use_testing_loop=None, use_prediction_loop=None, additional_metrics=None):
        model.visual_metrics = dict()

        model_util.enable_loop(model, "train", additional_metrics)
        if use_val_loop: model_util.enable_loop(model, "val", additional_metrics)
        if use_testing_loop: model_util.enable_loop(model, "test", additional_metrics)
        if use_prediction_loop: model_util.enable_loop(model, "pred", additional_metrics)

        if log_lr: 
            model.visual_metrics["Relative LR"] = []
            model.graph_lr_multiplier = 1


    @staticmethod
    def log(model: pl.LightningModule, loss, binary_pred=None, binary_target=None, metric_type="train", additional_metrics=None):
        full_metric_string = model_util._get_loop_type(metric_type)

        filtered_numpy_target = binary_target.detach().cpu().numpy()
        filtered_numpy_binary_pred = binary_pred.detach().cpu().numpy()

        precision = precision_score(filtered_numpy_target, filtered_numpy_binary_pred, zero_division=0) # 
        accuracy = accuracy_score(filtered_numpy_target, filtered_numpy_binary_pred)
        
        model.log(f'{metric_type}_loss', loss, on_epoch=model_util.log_per_epoch)
        model.log(f'{metric_type}_precision', precision, on_epoch=model_util.log_per_epoch)
        model.log(f'{metric_type}_accuracy', accuracy, on_epoch=model_util.log_per_epoch)
        if additional_metrics != None:
            for metric in additional_metrics:
                 model.log(f'{metric_type}_{metric}', additional_metrics[metric], on_epoch=model_util.log_per_epoch)

        model.visual_metrics[f"{full_metric_string} loss"].append(loss.item())
        model.visual_metrics[f"{full_metric_string} precision"].append(precision)
        model.visual_metrics[f"{full_metric_string} accuracy"].append(accuracy)
        if additional_metrics != None:
            for metric in additional_metrics:
                 model.visual_metrics[f"{full_metric_string} {metric}"].append(additional_metrics[metric])

        #print(f"{full_metric_string} Batch {batch_idx} Shape: " + str(x.shape) + " and " + str(x_add.shape))


    @staticmethod
    def configure_optimizers(model: pl.LightningModule, use_optimizer, betas=None, weight_decay=None, use_LR_scheduler=False, LR_increase_phase_percentage=None, LR_scheduler_min=None, LR_scheduler_max=None, LR_scheduler_convergence=None, fixed_learning_rate=None):
        optimizer_params = {"params": model.parameters()}
        if use_LR_scheduler == False and fixed_learning_rate != None: optimizer_params["lr"] = fixed_learning_rate
        if betas != None and type(betas) == list and betas[0] != None and betas[1] != None: optimizer_params["betas"] = fixed_learning_rate
        if weight_decay != None: optimizer_params["weight_decay"] = weight_decay

        optimizer = None
        if use_optimizer == "SGD": optimizer = torch.optim.SGD(**optimizer_params)
        elif use_optimizer == "AdamW": optimizer = torch.optim.AdamW(**optimizer_params)
        elif use_optimizer == "Adamax": optimizer = torch.optim.Adamax(**optimizer_params)
        elif use_optimizer == "Adam": optimizer = torch.optim.Adam(**optimizer_params)
        else: raise ValueError("Model used invalid optimizer string.")

        scheduler_fully_defined = (LR_increase_phase_percentage != None and LR_scheduler_min != None and LR_scheduler_max != None and LR_scheduler_convergence != None)
        scheduler_partly_defined = (LR_increase_phase_percentage != None or LR_scheduler_min != None or LR_scheduler_max != None or LR_scheduler_convergence != None)
        if use_LR_scheduler == True and scheduler_fully_defined == False: raise ValueError("Model wants to use LR scheduler, but does not define all neccessary params.")
        elif use_LR_scheduler == False and scheduler_partly_defined == True: raise ValueError("Model defines LR scheduler params but does not use LR scheduler.")

        if model_util.batch_count_per_epoch == None and use_LR_scheduler: raise ValueError("The batch count per epoch has not been set. Some LR schedulers require this so deterime the LR curve.")

        if use_LR_scheduler == True: 
            max_lr = LR_scheduler_max 
            base_lr = LR_scheduler_min 
            last_lr = LR_scheduler_convergence 
            batch_count_per_epoch = model_util.batch_count_per_epoch + 1
            div_factor = max_lr/base_lr
            max_div_factor = base_lr/last_lr
            increase_phase_percentage = LR_increase_phase_percentage
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr, total_steps=(model.trainer.max_epochs * batch_count_per_epoch), pct_start=increase_phase_percentage, div_factor=div_factor, final_div_factor=max_div_factor)
            model.graph_lr_multiplier = 1/max_lr
            return [optimizer], [lr_scheduler]
        else:
            model.graph_lr_multiplier = 1
            return optimizer

    @staticmethod
    def batch_step(model: pl.LightningModule, log_LR, use_LR_scheduler):
        if log_LR and model.trainer: 
            model.visual_metrics["Relative LR"].append(model.trainer.optimizers[0].param_groups[0]["lr"] * model.graph_lr_multiplier)
        if use_LR_scheduler:
            model.lr_schedulers().step()

    @staticmethod
    def enable_loop(model, loop_string, additional_metrics=None):
        full_metric_string = model_util._get_loop_type(loop_string)

        model.visual_metrics[f"{full_metric_string} loss"] = []
        model.visual_metrics[f"{full_metric_string} precision"] = []
        model.visual_metrics[f"{full_metric_string} accuracy"] = []
        if additional_metrics != None:
            for metric in additional_metrics:
                model.visual_metrics[f"{full_metric_string} {metric}"] = []

    @staticmethod
    def set_batch_count(batch_count):
        model_util.batch_count_per_epoch = batch_count

    @staticmethod
    def _get_loop_type(type_string):
        full_metric_string = ""
        if type_string == "train": full_metric_string = "Training"
        elif type_string == "val": full_metric_string = "Validation"
        elif type_string == "test": full_metric_string = "Testing"
        elif type_string == "predict": full_metric_string = "Prediction"
        else: raise ValueError("Model used invalid metric type for logging.")

        return full_metric_string

    # --- potential model features --- 

    @staticmethod
    def get_model_custom_pos_weights_support(model):
        if hasattr(model, model_util.model_pos_weight_function) and callable(getattr(model, model_util.model_pos_weight_function)):
            return True
        else:
            return False

    @staticmethod
    def set_model_custom_pos_weights(model, pos_weights):
        if hasattr(model, model_util.model_pos_weight_function) and callable(getattr(model, model_util.model_pos_weight_function)):
            change_func = getattr(model, model_util.model_pos_weight_function)
            change_func(pos_weights)
        else:
            raise RuntimeError("Model does not support the requested functionality.")


    @staticmethod
    def get_model_mask_support(model):
        if hasattr(model, model_util.model_mask_variable):
            return True
        else:
            return False

    @staticmethod
    def set_model_mask(model, mask_type):
        if hasattr(model, model_util.model_mask_variable):
            setattr(model, model_util.model_mask_variable, mask_type)
        else:
            raise RuntimeError("Model does not support the requested functionality.")

    @staticmethod
    def get_model_variable_prediction_length_support(model):
        if hasattr(model, model_util.model_prediction_length_variable):
            return True
        else:
            return False

    @staticmethod
    def set_model_prediction_length(model, prediction_length):
        if hasattr(model, model_util.model_prediction_length_variable):
            setattr(model, model_util.model_prediction_length_variable, prediction_length)
        else:
            raise RuntimeError("Model does not support the requested functionality.")
    
    @staticmethod
    def get_model_prediction_step_length(model):
        if hasattr(model, model_util.model_prediction_step_length_variable):
            return getattr(model, model_util.model_prediction_step_length_variable)
        else:
            raise RuntimeError("Model does not support the requested functionality.")

    @staticmethod
    def get_model_data_cut(model):
        data_cut_value = model.hparams.get(model_util.model_data_cut_hparam, None)
        return data_cut_value

    @staticmethod
    def set_model_desired_input_length(model, desired_input_length):
        if hasattr(model, model_util.model_input_length_variable):
            setattr(model, model_util.model_input_length_variable, desired_input_length)


    # --- functions for all models ---
    @staticmethod
    def standardize_output(output, sequence_length):
        output_shape = output.shape
        if output_shape[1] != sequence_length: # only occurs for obsolete flattening transformer
            output = output.view(-1, sequence_length)
        if output_shape[-1] == 1: # occurs for transformer based models
            output = output.squeeze(-1)

        return output
    
    @staticmethod
    def cut_labels(output, labels, sequence_length, prediction_split_count):
        if prediction_split_count > 1 and output.size(-2) != labels.size(-2): # occurs when performing multiple predictions for the same user for all models that return fewer output then inputs
            import math
            split_count = math.ceil(labels.size(-2)/sequence_length)
            return_seq_len = prediction_split_count/split_count
            assert return_seq_len == float(int(return_seq_len))
            return_seq_len = int(return_seq_len)
            new_label_tensor = None
            for seq_index in range(0, split_count):
                start_seq_index = -(seq_index*sequence_length) - return_seq_len
                end_seq_index = -(seq_index*sequence_length)

                i_partial_result = None
                if end_seq_index == 0: i_partial_result = label_tensor[..., start_seq_index:, :]
                else: i_partial_result = label_tensor[..., start_seq_index:end_seq_index:, :]

                if new_label_tensor == None: new_label_tensor = i_partial_result
                else: new_label_tensor = torch.cat((new_label_tensor, i_partial_result), -2)
            
            label_tensor = new_label_tensor
            if label_tensor.size() != labels.size(): raise ValueError(f"Failed to adjust labels. New label shape: {label_tensor.size()}")
        return output, labels


    # --- test outputs

    @staticmethod
    def log_batch_test_label_full(batch_type, batch, batch_idx):
        print(f" --- {batch_type} Batch {batch_idx} ---")
        for seq_id in range(batch[-1].size(0)):
            print(f"Sequence {seq_id}:")
            for label_index, label_tensor in enumerate(batch[-1][seq_id].flatten(0, 1)):
                print(f"Timeslot {label_index}:  {label_tensor.item()}")

    @staticmethod
    def log_batch_test_input_summary(batch_type, batch, batch_idx):
        print(f" --- {batch_type} Batch {batch_idx} ---")
        for seq_id in range(batch[0].size(0)):
            user = batch[0][seq_id][-1][-1][1].item()
            start = (batch[0][seq_id][0][0][0].item() / 2) + 1
            end = (batch[0][seq_id][-1][-1][0].item() + 1) / 2
            print(f"User {user} from {start} to {end}")

    @staticmethod
    def log_batch_test_nan_summary(batch_type, batch, batch_idx):
        print(f" --- {batch_type} Batch {batch_idx} ---")
        for seq_id in range(batch[-1].size(0)):
            
            first_input_index = None
            first_input_value = None
            first_label_index = None
            first_label_value = None
            for label_index, label_tensor in enumerate(batch[-1][seq_id].flatten(0, 1)):
                if not torch.isnan(label_tensor): 
                    first_label_index = label_index
                    first_label_value = f"{label_tensor}"
                    break

            
            for input_index, input_tensor in enumerate(batch[0][seq_id].flatten(0, 1)):
                if torch.all(input_tensor != 0): 
                    first_input_index = input_index
                    first_input_value = f"{input_tensor}"
                    break

            print(f"First values for sequence {seq_id}:  {first_input_index} / {first_label_index} with values {first_input_value} / {first_label_value}")
        
    
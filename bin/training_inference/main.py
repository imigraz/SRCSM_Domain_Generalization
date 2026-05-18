#!/usr/bin/python
import multiprocessing
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.mixed_precision as mixed_precision

import utils.io.image
from tensorflow_train_v2.dataset.dataset_iterator import DatasetIterator
from tensorflow_train_v2.losses.semantic_segmentation_losses import generalized_dice_loss
from tensorflow_train_v2.train_loop import MainLoopBase
import utils.sitk_image
from tensorflow_train_v2.utils.loss_metric_logger import LossMetricLogger
from tensorflow_train_v2.utils.output_folder_handler import OutputFolderHandler
from tensorflow_train_v2.utils.tensorflow_util import get_variable_and_ema_values, set_variable_values
from utils.segmentation.segmentation_test import SegmentationTest
from utils.segmentation.segmentation_statistics import SegmentationStatistics
from utils.segmentation.metrics import DiceMetric, SurfaceDistanceMetric
from dataset import Dataset
from network import Unet, UnetAvgLinear3D, RCNet
import os
import socket
import SimpleITK as sitk
from future_cached_dataset_iterator import FutureCachedDatasetIterator


def call_get_label_and_write_image(prediction, path, segmentation_test, input_img, transformation, image_spacing):
    prediction_labels = segmentation_test.get_label_image(prediction, input_img, image_spacing, transformation)
    utils.io.image.write(prediction_labels, path)
    return prediction_labels


class MainLoop(MainLoopBase):
    def __init__(self,
                 training_parameters,
                 dataset_parameters,
                 network_parameters,
                 loss_parameters,
                 experiment_parameters,
                 output_folder_name='',
                 is_inference=False):

        super().__init__()
        hostname = socket.gethostname()
        cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else None
        self.host_id = f'{hostname}_gpu:{cuda_visible_devices}'

        self.use_mixed_precision = True
        if self.use_mixed_precision:
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)

        self.training_parameters = training_parameters
        self.dataset_parameters = dataset_parameters
        self.network_parameters = network_parameters
        self.loss_parameters = loss_parameters
        self.experiment_parameters = experiment_parameters

        self.modality = self.training_parameters['modality']
        self.cv = self.training_parameters['cv']
        self.batch_size = 1
        self.learning_rate = self.training_parameters['learning_rate']
        self.lr_decay = self.training_parameters['lr_decay']
        self.max_iter = self.training_parameters['max_iter']
        self.test_iter = self.training_parameters['test_iter']
        self.disp_iter = 10
        self.snapshot_iter = self.test_iter
        self.test_initialization = self.training_parameters['test_initialization']
        self.current_iter = 0
        self.reg_constant = 0.000001
        self.data_format = 'channels_first'
        self.channel_axis = 1 if self.data_format == 'channels_first' else -1
        self.padding = 'same'
        self.output_folder_name = output_folder_name
        self.is_inference = is_inference
        self.map_predictions_to_ventricles_only = False if 'map_predictions_to_ventricles_only' not in experiment_parameters else experiment_parameters['map_predictions_to_ventricles_only']

        self.map_predictions_to_ventricles_only_dict = {
            1: 1,
            5: 2,
            2: 3,
        }

        if self.is_inference:
            # Note: parameters have no influence in inference
            self.use_randnet = False
            self.use_randnet_per_label = False
            self.use_randnet_per_label_smoothing = False
        else:
            self.use_randnet = training_parameters['use_randnet']
            self.use_randnet_per_label = training_parameters['use_randnet_per_label']
            self.use_randnet_per_label_smoothing = training_parameters['use_randnet_per_label_smoothing']

        self.use_ema_for_testing = self.training_parameters['use_ema_for_testing']

        if 'has_validation_groundtruth' in self.dataset_parameters:
            self.has_validation_groundtruth = self.dataset_parameters['has_validation_groundtruth']
        else:
            self.has_validation_groundtruth = self.cv != 0
            self.dataset_parameters.update(dict(has_validation_groundtruth=self.has_validation_groundtruth))

        self.local_base_folder = self.experiment_parameters['base_dataset_folder']
        self.base_output_folder = self.experiment_parameters['base_output_folder']

        self.test_output_process_pool_executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())

        self.num_labels = self.experiment_parameters['num_labels']

        self.image_size = self.experiment_parameters['image_size']
        self.image_extent = self.experiment_parameters['image_extent']

        self.image_spacing = [extent / size for extent, size in zip(self.image_extent, self.image_size)]

        self.unet = UnetAvgLinear3D
        self.network_parameters = dict(
            num_labels=self.num_labels,
            actual_network=self.unet,
            padding=self.padding,
            data_format=self.data_format,
            **network_parameters
        )

        self.dataset_parameters = dict(
            base_folder=self.local_base_folder,
            image_size=list(reversed(self.image_size)),
            image_spacing=list(reversed(self.image_spacing)),
            cv=self.cv,
            modality=self.modality,
            num_labels=self.num_labels,
            data_format=self.data_format,
            image_pixel_type=np.float16 if mixed_precision else np.float32,
            save_debug_images=False,
            **dataset_parameters
        )

        if not self.is_inference:
            self.metric_names = OrderedDict([(name, ['mean_{}'.format(name)] + list(map(lambda x: '{}_{}'.format(name, x), range(1, self.num_labels)))) for name in ['dice']])
        else:
            self.metric_names = OrderedDict([(name, ['mean_{}'.format(name)] + list(map(lambda x: '{}_{}'.format(name, x), range(1, self.num_labels)))) for name in ['dice', 'sd_mean', 'sd_median', 'sd_std', 'sd_max']])

        self.load_model_filename = training_parameters.get('load_model_filename', None)
        if self.load_model_filename is not None:
            self.test_initialization = True
            self.current_iter = training_parameters['load_model_iter']  # leads to weird error when loading: 'ValueError: Cannot assign to variable local_appearance/unet/contracting0/kernel:0 due to variable shape (3, 3, 3, 1, 128) and value shape () are incompatible'

    def run(self):
        super(MainLoop, self).run()

    def init_all(self):
        """
        Init all objects. Calls abstract init_*() functions.
        """
        self.init_model()
        self.init_optimizer()
        self.init_output_folder_handler()
        self.init_checkpoint()
        self.init_checkpoint_manager()
        self.init_datasets()
        self.init_loggers()

    def init_model(self):
        self.ema = tf.train.ExponentialMovingAverage(decay=0.999)
        self.norm_moving_average = tf.Variable(10.0)
        self.model = Unet(**self.network_parameters)
        self.all_model_trainable_variables = [var for var in self.model.trainable_variables]

    def gaussian_kernel_3d(self, size, sigma):
        """Creates a 3D Gaussian kernel."""
        coords = tf.range(-size // 2 + 1, size // 2 + 1, dtype=tf.float32)
        x, y, z = tf.meshgrid(coords, coords, coords)
        kernel = tf.exp(-(x ** 2 + y ** 2 + z ** 2) / (2.0 * sigma ** 2))
        kernel /= tf.reduce_sum(kernel)
        return kernel

    def apply_gaussian_smoothing_3d(self, volume, kernel_size=5, sigma=1.0):
        """
        Apply 3D Gaussian smoothing to a 3D volume using tf.nn.conv3d.
        """
        kernel = self.gaussian_kernel_3d(kernel_size, sigma)
        kernel = tf.expand_dims(kernel, axis=-1)
        kernel = tf.expand_dims(kernel, axis=-1)

        volume = tf.transpose(volume, perm=[0, 2, 3, 4, 1])
        smoothed_list = []
        for cur_label in range(self.num_labels):
            cur_smoothed = tf.nn.conv3d(volume[:,:,:,:,cur_label:cur_label+1], kernel, strides=[1, 1, 1, 1, 1], padding='SAME', data_format='NDHWC')
            smoothed_list.append(cur_smoothed)
        smoothed = tf.concat(smoothed_list, axis=-1)
        smoothed = tf.transpose(smoothed, perm=[0, 4, 1, 2, 3])
        return smoothed

    def get_smooth_per_label_mask(self, one_hot_labels):
        if self.batch_size != 1:
            print('this code snippet assumes batch_size == 1')
            exit(42)

        one_hot_labels_smooth = self.apply_gaussian_smoothing_3d(one_hot_labels, kernel_size=5, sigma=1.0)
        return one_hot_labels_smooth

    def apply_randnet_globally(self, image):
        image_augmented = None
        while (image_augmented is None or tf.reduce_sum(tf.cast(image_augmented, tf.float32)) == 0):
            self.randnet = RCNet()
            image_augmented = self.randnet(image)
        return image_augmented

    def apply_randnet_per_label(self, image, labels):
        one_hot_label_mask = self.split_labels_tf(labels, w_batch_dim=True)
        if self.use_randnet_per_label_smoothing:
            one_hot_label_mask = self.get_smooth_per_label_mask(one_hot_label_mask)

        image_augmented = tf.zeros_like(image)
        for cur_label_val in range(self.num_labels):
            cur_label_mask = tf.cast(one_hot_label_mask[:, cur_label_val:cur_label_val + 1], dtype=image.dtype)

            cur_image_augmented = None
            while (cur_image_augmented is None or tf.reduce_sum(tf.cast(cur_image_augmented, tf.float32)) == 0):
                self.randnet = RCNet()
                cur_image_augmented = self.randnet(image)

            image_augmented += cur_image_augmented * cur_label_mask

        return image_augmented

    def get_new_randnet_model_image(self, image, labels):

        if not self.use_randnet:
            image_augmented = image
        else:
            if not self.use_randnet_per_label:
                image_augmented = self.apply_randnet_globally(image)
            else:
                image_augmented = self.apply_randnet_per_label(image, labels)

        return image_augmented

    def save_model(self):
        """
        Save the model.
        """
        old_weights = [tf.keras.backend.get_value(var) for var in self.model.trainable_variables]
        new_weights = [tf.keras.backend.get_value(self.ema.average(var)) for var in self.model.trainable_variables]
        for var, weights in zip(self.model.trainable_variables, new_weights):
            tf.keras.backend.set_value(var, weights)
        super(MainLoop, self).save_model()
        for var, weights in zip(self.model.trainable_variables, old_weights):
            tf.keras.backend.set_value(var, weights)

    def init_optimizer(self):
        if self.lr_decay:
            self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.learning_rate, self.max_iter, 0.1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        if self.use_mixed_precision:
            self.optimizer = mixed_precision.LossScaleOptimizer(self.optimizer)

    def init_checkpoint(self):
        self.checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)

    def init_output_folder_handler(self):
        self.output_folder_handler = OutputFolderHandler(self.base_output_folder, model_name=self.model.name, additional_info=self.output_folder_name)

    def init_datasets(self):
        network_image_size_w_channel = [1] + self.image_size if self.data_format == 'channels_first' else self.image_size + [1]

        data_generator_entries = dict(
            image=network_image_size_w_channel,
            labels=network_image_size_w_channel,
        )
        data_generator_types = dict(
            image=tf.float16 if self.use_mixed_precision else tf.float32,
            labels=tf.uint8,
        )

        self.data_generator_entry_list = list(data_generator_entries.keys())

        self.dataset = Dataset(**self.dataset_parameters)

        if not self.is_inference:
            self.dataset_train = self.dataset.dataset_train()
            self.dataset_train_iter = DatasetIterator(dataset=self.dataset_train, data_names_and_shapes=data_generator_entries, data_types=data_generator_types, batch_size=self.batch_size)
            self.train_dataset_iterator_cache = FutureCachedDatasetIterator(self.dataset_train_iter, cache_size=5)

        self.dataset_val = self.dataset.dataset_val()


    def init_loggers(self):
        self.loss_metric_logger_train = LossMetricLogger('train',
                                                         self.output_folder_handler.path('train'),
                                                         self.output_folder_handler.path('train.csv'))
        self.loss_metric_logger_val = LossMetricLogger('test',
                                                       self.output_folder_handler.path('test'),
                                                       self.output_folder_handler.path('test.csv'))


    def split_labels_tf(self, labels, w_batch_dim):
        if w_batch_dim:
            axis = self.channel_axis
        else:
            # axis wo batch dimension
            axis = 0 if self.data_format == 'channels_first' else -1
        split_labels = tf.one_hot(tf.squeeze(labels, axis=axis), depth=self.num_labels, axis=axis)
        return split_labels


    @tf.function
    def call_model_and_loss(self, image, labels, training):
        prediction_list = []
        prediction, _, _, _, _ = self.model(image, training=training)
        prediction_list.append(prediction)

        losses_dict = {}

        loss_dice = self.supervised_losses(labels, prediction, **self.loss_parameters)
        losses_dict['loss_dice'] = loss_dice

        return prediction_list, losses_dict


    def train_step(self):
        dataset_entry = self.train_dataset_iterator_cache.get_next()

        training_data_list = list(dataset_entry)
        training_data_dict = {k: v for k, v in zip(self.data_generator_entry_list, training_data_list)}

        image = training_data_dict['image'] if 'image' in training_data_dict else None
        labels = training_data_dict['labels'] if 'labels' in training_data_dict else None

        nan_loss_counter = 0
        while (True):

            augmented_image = self.get_new_randnet_model_image(image, labels)

            valid = self.train_step_proper(augmented_image, labels)
            if valid:
                break

            if nan_loss_counter > 100:
                print(f'nan_loss_counter {nan_loss_counter} - exiting program')
                exit(42)

            nan_loss_counter += 1

        if nan_loss_counter > 0:
            print(f'nan_loss_counter {nan_loss_counter}')


    @tf.function
    def train_step_proper(self, augmented_image, labels):
        labels_split = self.split_labels_tf(labels, w_batch_dim=True)

        with tf.GradientTape() as tape:
            _, losses = self.call_model_and_loss(augmented_image, labels_split, training=True)

            new_losses = dict(loss_sum=tf.reduce_sum(list([v for k, v in losses.items()])))
            new_losses.update(losses)
            losses = new_losses

            if self.reg_constant > 0:
                losses['loss_reg'] = self.reg_constant * tf.reduce_sum(self.model.losses)
            loss = tf.reduce_sum(list(losses.values()))
            if self.use_mixed_precision:
                scaled_loss = self.optimizer.get_scaled_loss(loss)

        if tf.math.is_nan(loss):
            return False

        variables = self.model.trainable_weights
        metric_dict = losses
        clip_norm = self.norm_moving_average * 5
        if self.use_mixed_precision:
            scaled_grads = tape.gradient(scaled_loss, variables)
            grads = self.optimizer.get_unscaled_gradients(scaled_grads)
            grads, norm = tf.clip_by_global_norm(grads, clip_norm)
            loss_scale = self.optimizer.loss_scale
            metric_dict.update({'loss_scale': loss_scale})
        else:
            grads = tape.gradient(loss, variables)
            grads, norm = tf.clip_by_global_norm(grads, clip_norm)
        if tf.math.is_finite(norm):
            alpha = 0.01
            self.norm_moving_average.assign(alpha * tf.minimum(norm, clip_norm) + (1 - alpha) * self.norm_moving_average)
        metric_dict.update({'norm': norm, 'norm_average': self.norm_moving_average})
        self.optimizer.apply_gradients(zip(grads, variables))
        self.ema.apply(variables)

        self.loss_metric_logger_train.update_metrics(metric_dict)
        return True

    @tf.function
    def supervised_losses(self, mask, prediction, loss=None):
        loss_supervised = loss(labels=mask, logits=prediction, data_format=self.data_format)
        return loss_supervised

    def get_summary_dict(self, segmentation_statistics, name):
        mean_list = segmentation_statistics.get_metric_mean_list(name)
        mean_of_mean_list = np.mean(mean_list)
        return OrderedDict(list(zip(self.metric_names[name], [mean_of_mean_list] + mean_list)))

    def test(self):
        print('Testing...')

        if not self.first_iteration:
            if self.use_ema_for_testing:
                model_var_values, model_var_ema_values = get_variable_and_ema_values(self.all_model_trainable_variables, self.ema)
                set_variable_values(self.all_model_trainable_variables, model_var_ema_values)

        channel_axis = 0
        if self.data_format == 'channels_last':
            channel_axis = 3
        labels = list(range(self.num_labels))
        labels_wo_background = list(range(1, self.num_labels))
        segmentation_test = SegmentationTest(labels,
                                             channel_axis=channel_axis,
                                             interpolator='linear',
                                             largest_connected_component=False,
                                             all_labels_are_connected=False)

        if not self.map_predictions_to_ventricles_only:
            labels_wo_background_to_evaluate = labels_wo_background
        else:
            labels_wo_background_to_evaluate = labels_wo_background[:3]

        if not self.is_inference:
            metrics = OrderedDict([
                ('dice', DiceMetric()),
            ])
        else:
            metrics = OrderedDict([
                ('dice', DiceMetric()),
                (('sd_mean', 'sd_median', 'sd_std', 'sd_max'), SurfaceDistanceMetric()),
            ])
        segmentation_statistics = SegmentationStatistics(labels_wo_background_to_evaluate,
                                                         self.output_folder_handler.path_for_iteration(self.current_iter),
                                                         metrics=metrics)

        test_dataset_iterator_cache = FutureCachedDatasetIterator(self.dataset_val, cache_size=3)
        for _ in tqdm(range(test_dataset_iterator_cache.num_entries()), desc='Testing'):
            dataset_entry = test_dataset_iterator_cache.get_next()

            if self.has_validation_groundtruth:
                dataset_entry['generators']['labels'] = self.split_labels_tf(dataset_entry['generators']['labels'], w_batch_dim=False)
            current_id = dataset_entry['id']['image_id']
            datasources = dataset_entry['datasources']
            generators = dataset_entry['generators']
            transformations = dataset_entry['transformations']
            if self.has_validation_groundtruth:
                prediction_wo_activation_list, losses = self.call_model_and_loss(np.expand_dims(generators['image'], axis=0), np.expand_dims(generators['labels'], axis=0), False)
                prediction_wo_activation = prediction_wo_activation_list[0]
            else:
                prediction_wo_activation, _, _, _, _ = self.model(np.expand_dims(generators['image'], axis=0), False)

            prediction = np.squeeze(tf.nn.softmax(prediction_wo_activation, axis=1 if self.data_format == 'channels_first' else -1), axis=0)
            input_img = datasources['image']
            transformation = transformations['image']

            if not self.map_predictions_to_ventricles_only:
                future_prediction_labels = self.test_output_process_pool_executor.submit(call_get_label_and_write_image, **dict(prediction=prediction, path=self.output_folder_handler.path_for_iteration(self.current_iter, f'{current_id}.nii.gz'), segmentation_test=segmentation_test, input_img=input_img, transformation=transformation, image_spacing=self.image_spacing))
            else:
                future_prediction_labels = self.test_output_process_pool_executor.submit(call_get_label_and_write_image, **dict(prediction=prediction, path=self.output_folder_handler.path_for_iteration(self.current_iter, 'all_labels', f'{current_id}.nii.gz'), segmentation_test=segmentation_test, input_img=input_img, transformation=transformation, image_spacing=self.image_spacing))

            if self.has_validation_groundtruth:
                groundtruth = datasources['labels']
                prediction_labels = future_prediction_labels.result()

                if not self.map_predictions_to_ventricles_only:
                    segmentation_statistics.add_labels(current_id, prediction_labels, groundtruth)
                else:
                    prediction_labels_np = sitk.GetArrayFromImage(prediction_labels)
                    prediction_labels_ventricles_only_np = np.zeros_like(prediction_labels_np)
                    for old, new in self.map_predictions_to_ventricles_only_dict.items():
                        prediction_labels_ventricles_only_np = np.where(prediction_labels_np == old, new, prediction_labels_ventricles_only_np)
                    prediction_labels_ventricles_only_np = prediction_labels_ventricles_only_np.astype(prediction_labels_np.dtype)
                    prediction_labels_ventricles_only = sitk.GetImageFromArray(prediction_labels_ventricles_only_np)
                    prediction_labels_ventricles_only.CopyInformation(prediction_labels)
                    utils.io.image.write(prediction_labels_ventricles_only, self.output_folder_handler.path_for_iteration(self.current_iter, 'cine_labels',f'{current_id}_ventricles.nrrd'))
                    segmentation_statistics.add_labels(current_id, prediction_labels_ventricles_only, groundtruth)

        if self.has_validation_groundtruth:
            segmentation_statistics.finalize()
            summary_values = OrderedDict()
            for name in self.metric_names.keys():
                summary_values.update(self.get_summary_dict(segmentation_statistics, name))
            self.loss_metric_logger_val.update_metrics(summary_values)

        self.loss_metric_logger_val.finalize(self.current_iter)

        if not self.first_iteration:
            if self.use_ema_for_testing:
                set_variable_values(self.all_model_trainable_variables, model_var_values)


def run(
        modality,
        cur_cv,
        setup,
        base_dataset_folder,
        base_output_folder,
        num_labels,
        image_size,
        image_extent,
        use_randnet,
        use_randnet_per_label,
        use_randnet_per_label_smoothing,
):

    training_parameters = dict(
        modality=modality,
        cv=cur_cv,
        learning_rate=0.00005,
        lr_decay=False,
        max_iter=50000,
        test_iter=10000,
        test_initialization=False,
        image_size_per_dim=image_size[0],
        use_ema_for_testing=True,
        use_randnet=use_randnet,
        use_randnet_per_label=use_randnet_per_label,
        use_randnet_per_label_smoothing=use_randnet_per_label_smoothing,
    )

    dataset_parameters = dict(
        input_gaussian_sigma=1.0,
        label_gaussian_sigma=1.0,
        setup_folder_to_use=setup,
        cached_datasource=True,
        ct_random_shift=0.2,
        ct_random_scale=0.2,
        ct_clamp_min=-1.0,
        ct_clamp_max=1.0,
        mr_random_shift=0.2,
        mr_random_scale=0.4,
        mr_clamp_min=-1.0,
        mr_clamp_max=None,
    )

    loss_parameters = dict(
        loss=generalized_dice_loss,
    )

    network_parameters = dict(
        num_filters_base=64,
        num_levels=5,
        dropout_ratio=0.1,
        activation='lrelu',
    )

    experiment_parameters = dict(
        base_dataset_folder=base_dataset_folder,
        base_output_folder=base_output_folder,
        num_labels=num_labels,
        image_size=image_size,
        image_extent=image_extent,
    )

    if 'load_model_base' in locals() and 'load_model_iter' in locals():
        load_model_base = locals()['load_model_base']
        load_model_iter = locals()['load_model_iter']
        training_parameters['load_model_filename'] = os.path.join(load_model_base, 'weights', f'ckpt-{load_model_iter}')
        training_parameters['load_model_iter'] = load_model_iter
    else:
        training_parameters['load_model_filename'] = None
        training_parameters['load_model_iter'] = None

    output_folder_name = f"default" + '/'
    output_folder_name += f"{modality}" + '/'
    output_folder_name += f"cv{cur_cv}" + '/'
    output_folder_name += f"{dataset_parameters['setup_folder_to_use']}" + '/'
    output_folder_name += f"{training_parameters['image_size_per_dim']}" + '/'

    output_folder_name += f'unet'

    output_folder_name += f"_RCNet" if training_parameters['use_randnet'] else ""
    output_folder_name += f"PLab" if training_parameters['use_randnet'] and training_parameters['use_randnet_per_label'] else ""
    output_folder_name += f"Sm" if training_parameters['use_randnet'] and training_parameters['use_randnet_per_label'] and training_parameters['use_randnet_per_label_smoothing'] else ""

    output_folder_name += f"_it{training_parameters['max_iter']}"
    output_folder_name += f"_lr{training_parameters['learning_rate']}"

    loop = MainLoop(
        training_parameters=training_parameters,
        dataset_parameters=dataset_parameters,
        network_parameters=network_parameters,
        loss_parameters=loss_parameters,
        experiment_parameters=experiment_parameters,
        output_folder_name=output_folder_name,
    )
    loop.run()
    # del loop


def run_test(
        cur_cv,
        setup,
        base_dataset_folder,
        base_output_folder,
        num_labels,
        image_size,
        image_extent,
        is_inference,
        params_dict,
        wrapper_dict,
):
    load_model_base = params_dict['load_model_base']
    load_model_iter = params_dict['load_model_iter']
    final_output_folder_in_path = params_dict['final_output_folder_in_path']

    additional_dataset_parameters = wrapper_dict['additional_dataset_parameters']
    has_validation_groundtruth = wrapper_dict['has_validation_groundtruth']
    dataset_name_train = wrapper_dict['dataset_name_train']
    dataset_name_test = wrapper_dict['dataset_name_test']
    modality = wrapper_dict['modality']
    map_predictions_to_ventricles_only = wrapper_dict['map_predictions_to_ventricles_only']

    training_parameters = dict(
        modality=modality,
        cv=cur_cv,
        learning_rate=0.00005,
        lr_decay=False,
        max_iter=50000,
        test_iter=10000,
        test_initialization=False,
        image_size_per_dim=image_size[0],
        use_ema_for_testing=True,
    )

    dataset_parameters = dict(
        input_gaussian_sigma=1.0,
        label_gaussian_sigma=1.0,
        setup_folder_to_use=setup,
        cached_datasource=True,
        ct_random_shift=0.2,
        ct_random_scale=0.2,
        ct_clamp_min=-1.0,
        ct_clamp_max=1.0,
        mr_random_shift=0.2,
        mr_random_scale=0.4,
        mr_clamp_min=-1.0,
        mr_clamp_max=None,
        has_validation_groundtruth=has_validation_groundtruth,
    )
    dataset_parameters.update(additional_dataset_parameters)

    loss_parameters = dict(
        loss=generalized_dice_loss,
    )

    network_parameters = dict(
        num_filters_base=64,
        num_levels=5,
        dropout_ratio=0.1,
        activation='lrelu',
    )

    experiment_parameters = dict(
        base_dataset_folder=base_dataset_folder,
        base_output_folder=base_output_folder,
        num_labels=num_labels,
        image_size=image_size,
        image_extent=image_extent,
        map_predictions_to_ventricles_only=map_predictions_to_ventricles_only,
    )

    if 'load_model_base' in locals() and 'load_model_iter' in locals():
        load_model_base = locals()['load_model_base']
        load_model_iter = locals()['load_model_iter']
        training_parameters['load_model_filename'] = os.path.join(load_model_base, 'weights', f'ckpt-{load_model_iter}')
        training_parameters['load_model_iter'] = load_model_iter
    else:
        training_parameters['load_model_filename'] = None
        training_parameters['load_model_iter'] = None


    output_folder_name = f"inference"
    if cur_cv == 1:
        output_folder_name += '_in_domain_comparison'

    output_folder_name += '/'

    output_folder_name += f'{dataset_name_train}__to__{dataset_name_test}' + '/'
    output_folder_name += additional_dataset_parameters['image_base_folder'] + '/'
    output_folder_name += dataset_parameters['setup_folder_to_use'] + '/'
    output_folder_name += f"{training_parameters['image_size_per_dim']}" + '/'

    path_tmp = load_model_base.split('/')
    output_folder_name += path_tmp[-2] if path_tmp[0] == '' else path_tmp[-1]

    output_folder_name += f'_loadModel' if 'load_model_filename' in training_parameters and training_parameters['load_model_filename'] != None else ''

    output_folder_name += '/'
    output_folder_name += final_output_folder_in_path + '/'

    loop = MainLoop(
        training_parameters=training_parameters,
        dataset_parameters=dataset_parameters,
        network_parameters=network_parameters,
        loss_parameters=loss_parameters,
        experiment_parameters=experiment_parameters,
        output_folder_name=output_folder_name,
        is_inference=is_inference,
    )
    loop.run_test()

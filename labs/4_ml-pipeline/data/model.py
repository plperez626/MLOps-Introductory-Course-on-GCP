# Copyright 2021 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Covertype Keras WideDeep Classifier."""


import functools
import absl
import os
from typing import List, Text

import kerastuner
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_transform as tft
from tensorflow_transform.tf_metadata import schema_utils

from tfx.components.trainer.executor import TrainerFnArgs
from tfx.components.trainer.fn_args_utils import DataAccessor
from tfx.components.tuner.component import TunerFnResult
from tfx_bsl.tfxio import dataset_options

import features

EPOCHS = 1
TRAIN_BATCH_SIZE = 64
EVAL_BATCH_SIZE = 64


def _gzip_reader_fn(filenames):
  """Small utility returning a record reader that can read gzip'ed files."""
  return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _get_serve_tf_examples_fn(model, tf_transform_output):
  """Returns a function that parses a serialized tf.Example and applies TFT."""

  model.tft_layer = tf_transform_output.transform_features_layer()

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    """Returns the output to be used in the serving signature."""
    feature_spec = tf_transform_output.raw_feature_spec()
    feature_spec.pop(features.LABEL_KEY)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

    transformed_features = model.tft_layer(parsed_features)

    return model(transformed_features)

  return serve_tf_examples_fn
# def _get_serve_tf_examples_fn(model, tf_transform_output):
#     """Returns a function that parses a serialized tf.Example and applies TFT."""

#     # Add the TFT layer to the model
#     model.tft_layer = tf_transform_output.transform_features_layer()

#     @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
#     def serve_tf_examples_fn(serialized_tf_examples):
#         """Returns the output to be used in the serving signature."""
#         # Get the feature spec and remove the label key (if present)
#         feature_spec = tf_transform_output.raw_feature_spec()
#         if features.LABEL_KEY in feature_spec:
#             feature_spec.pop(features.LABEL_KEY)

#         # Parse the serialized examples
#         parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

#         # Convert sparse tensors to dense tensors
#         dense_features = {}
#         for key, value in parsed_features.items():
#             if isinstance(value, tf.SparseTensor):
#                 dense_features[key] = tf.sparse.to_dense(value)
#             else:
#                 dense_features[key] = value

#         # Debugging: Print dense features to verify their structure
#         print("Dense Features:", dense_features)

#         # Apply the TFT preprocessing layer
#         transformed_features = model.tft_layer(dense_features)

#         # Debugging: Print transformed features to verify their structure
#         print("Transformed Features:", transformed_features)

#         # Call the model with the transformed features
#         return model(transformed_features)

#     return serve_tf_examples_fn


def _input_fn(file_pattern: List[Text],
              data_accessor: DataAccessor,
              tf_transform_output: tft.TFTransformOutput,
              batch_size: int = 200) -> tf.data.Dataset:
  """Generates features and label for tuning/training.

  Args:
    file_pattern: List of paths or patterns of input tfrecord files.
    data_accessor: DataAccessor for converting input to RecordBatch.
    tf_transform_output: A TFTransformOutput.
    batch_size: representing the number of consecutive elements of returned
      dataset to combine in a single batch

  Returns:
    A dataset that contains (features, indices) tuple where features is a
      dictionary of Tensors, and indices is a single Tensor of label indices.
  """
  dataset = data_accessor.tf_dataset_factory(
      file_pattern,
      dataset_options.TensorFlowDatasetOptions(
          batch_size=batch_size, label_key=features.transformed_name(features.LABEL_KEY)),
      tf_transform_output.transformed_metadata.schema)
    
  return dataset


def _get_hyperparameters() -> kerastuner.HyperParameters:
  """Returns hyperparameters for building Keras model."""
  hp = kerastuner.HyperParameters()
  # Defines search space.
  hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4], default=1e-3)
  hp.Int('n_layers', 1, 2, default=1)
  with hp.conditional_scope('n_layers', 1):
        hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=8)
  with hp.conditional_scope('n_layers', 2):
        hp.Int('n_units_1', min_value=8, max_value=128, step=8, default=8)
        hp.Int('n_units_2', min_value=8, max_value=128, step=8, default=8)        

  return hp


def build_keras_model(hparams: kerastuner.HyperParameters, 
                      tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:
  # Define inputs for numeric features
  numeric_inputs = {
      features.transformed_name(key): tf.keras.layers.Input(name=features.transformed_name(key), shape=(1,), dtype=tf.float32)
      for key in features.NUMERIC_FEATURE_KEYS
  }
  
  # Define inputs for categorical features
  categorical_inputs = {
      features.transformed_name(key): tf.keras.layers.Input(name=features.transformed_name(key), shape=(1,), dtype=tf.int32)
      for key in features.CATEGORICAL_FEATURE_KEYS
  }
  
  # Combine all inputs
  all_inputs = {**numeric_inputs, **categorical_inputs}
  
  # Process numeric features (deep part)
  numeric_features = tf.keras.layers.concatenate(list(numeric_inputs.values()))
  
  # Process categorical features (wide part)
  categorical_features_list = []
  for key, input_tensor in categorical_inputs.items():
      num_buckets = tf_transform_output.num_buckets_for_transformed_feature(key)
      embedding = tf.keras.layers.Embedding(num_buckets, 8)(input_tensor)
      embedding = tf.keras.layers.Flatten()(embedding)
      categorical_features_list.append(embedding)
  
  if categorical_features_list:
      categorical_features = tf.keras.layers.concatenate(categorical_features_list)
  else:
      categorical_features = tf.keras.layers.Lambda(lambda x: x)(numeric_features)  # Just a passthrough
  
  # Deep path
  deep = numeric_features
  for n in range(int(hparams.get('n_layers'))):
      deep = tf.keras.layers.Dense(units=hparams.get('n_units_' + str(n + 1)))(deep)
  
  # Combine wide and deep paths
  combined = tf.keras.layers.concatenate([deep, categorical_features])
  
  # Output layer
  output = tf.keras.layers.Dense(features.NUM_CLASSES, activation='softmax')(combined)
  
  # Create model
  model = tf.keras.Model(inputs=all_inputs, outputs=output)
  
  # Compile model with Keras 3 compatible optimizer
  model.compile(
      loss='sparse_categorical_crossentropy',
      optimizer=tf.keras.optimizers.Adam(learning_rate=hparams.get('learning_rate')),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
  )
  
  model.summary(print_fn=absl.logging.info)
  return model

# def _build_keras_model(hparams: kerastuner.HyperParameters, 
#                      tf_transform_output: tft.TFTransformOutput) -> tf.keras.Model:
#     """Creates a Keras WideDeep Classifier model.
#     Args:
#       hparams: Holds HyperParameters for tuning.
#       tf_transform_output: A TFTransformOutput.
#     Returns:
#       A keras Model.
#     """
#     # Get all transformed feature names
#     numeric_feature_names = [features.transformed_name(key) for key in features.NUMERIC_FEATURE_KEYS]
#     categorical_feature_names = [features.transformed_name(key) for key in features.CATEGORICAL_FEATURE_KEYS]
#     all_feature_names = numeric_feature_names + categorical_feature_names
    
#     # Create separate inputs for each feature
#     inputs = {}
#     for feature_name in numeric_feature_names:
#         inputs[feature_name] = tf.keras.layers.Input(
#             shape=(1,), name=feature_name, dtype=tf.float32)
    
#     for feature_name in categorical_feature_names:
#         inputs[feature_name] = tf.keras.layers.Input(
#             shape=(1,), name=feature_name, dtype=tf.int32)
    
#     # Process numeric features (deep part)
#     numeric_tensors = [inputs[name] for name in numeric_feature_names]
    
#     if numeric_tensors:
#         deep = tf.keras.layers.concatenate(numeric_tensors)
#     else:
#         deep = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(list(inputs.values())[0])
    
#     # Add dense layers for deep part
#     for n in range(int(hparams.get('n_layers'))):
#         deep = tf.keras.layers.Dense(
#             units=hparams.get('n_units_' + str(n + 1)),
#             activation='relu')(deep)
    
#     # Process categorical features (wide part)
#     wide_tensors = []
#     for name in categorical_feature_names:
#         # Get the number of buckets for this feature
#         num_buckets = tf_transform_output.num_buckets_for_transformed_feature(name)
#         # Use embedding followed by flatten as an alternative to one-hot encoding
#         embedding = tf.keras.layers.Embedding(
#             input_dim=num_buckets,
#             output_dim=num_buckets)(tf.cast(inputs[name], tf.int32))
#         wide_tensors.append(tf.keras.layers.Flatten()(embedding))
    
#     if wide_tensors:
#         wide = tf.keras.layers.concatenate(wide_tensors)
#     else:
#         wide = tf.keras.layers.Lambda(lambda x: tf.zeros_like(x))(deep)
    
#     # Combine wide and deep parts
#     combined = tf.keras.layers.concatenate([deep, wide])
    
#     # Output layer
#     output = tf.keras.layers.Dense(
#         features.NUM_CLASSES, activation='softmax')(combined)
    
#     # Create model
#     model = tf.keras.Model(inputs=inputs, outputs=output)
    
#     # Compile model
#     model.compile(
#         loss='sparse_categorical_crossentropy',
#         optimizer=tf.keras.optimizers.Adam(lr=hparams.get('learning_rate')),
#         metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
#     model.summary(print_fn=absl.logging.info)
#     return model

# TFX Tuner will call this function.
def tuner_fn(fn_args: TrainerFnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  transform_graph = tft.TFTransformOutput(fn_args.transform_graph_path)
  
  # Construct a build_keras_model_fn that just takes hyperparams from get_hyperparameters as input.
  build_keras_model_fn = functools.partial(
      build_keras_model, tf_transform_output=transform_graph)  

  # BayesianOptimization is a subclass of kerastuner.Tuner which inherits from BaseTuner.    
  tuner = kerastuner.BayesianOptimization(
      build_keras_model_fn,
      max_trials=5,
      hyperparameters=_get_hyperparameters(),
      # New entries allowed for n_units hyperparameter construction conditional on n_layers selected.
#       allow_new_entries=True,
#       tune_new_entries=True,
      objective=kerastuner.Objective('val_sparse_categorical_accuracy', 'max'),
      directory=fn_args.working_dir,
      project_name='covertype_tuning')
  
  train_dataset = _input_fn(
      fn_args.train_files,
      fn_args.data_accessor,
      transform_graph,
      batch_size=TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files,
      fn_args.data_accessor,
      transform_graph,
      batch_size=EVAL_BATCH_SIZE)

  return TunerFnResult(
      tuner=tuner,
      fit_kwargs={
          'x': train_dataset,
          'validation_data': eval_dataset,
          'steps_per_epoch': fn_args.train_steps,
          'validation_steps': fn_args.eval_steps
      })


# TFX Trainer will call this function.
def run_fn(fn_args: TrainerFnArgs):
  """Train the model based on given args.
  Args:
    fn_args: Holds args used to train the model as name/value pairs.
  """

  tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

  train_dataset = _input_fn(
      fn_args.train_files, 
      fn_args.data_accessor, 
      tf_transform_output, 
      TRAIN_BATCH_SIZE)

  eval_dataset = _input_fn(
      fn_args.eval_files, 
      fn_args.data_accessor,
      tf_transform_output, 
      EVAL_BATCH_SIZE)

  if fn_args.hyperparameters:
    hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
  else:
    # This is a shown case when hyperparameters is decided and Tuner is removed
    # from the pipeline. User can also inline the hyperparameters directly in
    # _build_keras_model.
    hparams = _get_hyperparameters()
  absl.logging.info('HyperParameters for training: %s' % hparams.get_config())
  
  # Distribute training over multiple replicas on the same machine.
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
        model = build_keras_model(
            hparams=hparams,
            tf_transform_output=tf_transform_output)

  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=fn_args.model_run_dir, update_freq='batch')

  model.fit(
      train_dataset,
      epochs=EPOCHS,
      steps_per_epoch=fn_args.train_steps,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])
    
  signatures = {
      'serving_default':
          _get_serve_tf_examples_fn(model,
                                    tf_transform_output).get_concrete_function(
                                        tf.TensorSpec(
                                            shape=[None],
                                            dtype=tf.string,
                                            name='examples')),
  }
  # model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
  # absl.logging.info("Model saved successfully") 

#   @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
#   def serve_tf_examples_fn(examples):
#     # Preprocess the examples using tf_transform_output
#         print("Feature Spec:", tf_transform_output.raw_feature_spec())
#         processed_features = tf_transform_output.transform_raw_features(examples)

#         # Call the model
#         return model(processed_features)
  #serve_tf_examples_fn = _get_serve_tf_examples_fn(model, tf_transform_output)

    # Save the model with the serving signature
  tf.saved_model.save(
        model,
        export_dir=fn_args.serving_model_dir,
        signatures=signatures
            #'serving_default': serve_tf_examples_fn,
        
        )
  absl.logging.info("Model saved successfully")

# def run_fn(fn_args: TrainerFnArgs):
#   """Train the model based on given args.
#   Args:
#     fn_args: Holds args used to train the model as name/value pairs.
#   """
#   tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
#   train_dataset = _input_fn(
#       fn_args.train_files, 
#       fn_args.data_accessor, 
#       tf_transform_output, 
#       TRAIN_BATCH_SIZE)
#   eval_dataset = _input_fn(
#       fn_args.eval_files, 
#       fn_args.data_accessor,
#       tf_transform_output, 
#       EVAL_BATCH_SIZE)
  
#   if fn_args.hyperparameters:
#     hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
#   else:
#     # This is a shown case when hyperparameters is decided and Tuner is removed
#     # from the pipeline. User can also inline the hyperparameters directly in
#     # build_keras_model.
#     hparams = _get_hyperparameters()
#   absl.logging.info('HyperParameters for training: %s' % hparams.get_config())
  
#   # Distribute training over multiple replicas on the same machine.
#   mirrored_strategy = tf.distribute.MirroredStrategy()
#   with mirrored_strategy.scope():
#         model = build_keras_model(
#             hparams=hparams,
#             tf_transform_output=tf_transform_output)
            
#   tensorboard_callback = tf.keras.callbacks.TensorBoard(
#       log_dir=fn_args.model_run_dir, update_freq='batch')
      
#   model.fit(
#       train_dataset,
#       epochs=EPOCHS,
#       steps_per_epoch=fn_args.train_steps,
#       validation_data=eval_dataset,
#       validation_steps=fn_args.eval_steps,
#       callbacks=[tensorboard_callback])
    
#   # For Keras 3 compatibility, we need a different approach to signatures
#   # First, save the model in Keras format
#   model_path = fn_args.serving_model_dir
#   # model.save(model_path + '.keras')
#   serving_model_dir = fn_args.serving_model_dir
  
#   # Create an empty directory if it doesn't exist
#   if not tf.io.gfile.exists(serving_model_dir):
#     tf.io.gfile.makedirs(serving_model_dir)
  
#   # Create a basic saved model in the correct directory
#   tf.saved_model.save(model, serving_model_dir)
  
#   # Log success
#   absl.logging.info(f"Model saved to {serving_model_dir}")
  
# #   # Then, create and save a TensorFlow SavedModel with serving signature
# #   serving_fn = _get_serve_tf_examples_fn(model, tf_transform_output)
  
# #   # Convert to a concrete function
# #   concrete_serving_fn = serving_fn.get_concrete_function(
# #       tf.TensorSpec(shape=[None], dtype=tf.string, name='examples'))
  
# #   # Create a new SavedModel with just the serving function
# #   model.save(fn_args.serving_model_dir+'.keras')
# #   tf.saved_model.save(
# #       obj=tf.Module(),
# #       export_dir=model_path,
# #       signatures={
# #           'serving_default': concrete_serving_fn
# #       }
#  # )
  
#   # Optionally, you can log that the model has been saved
#   absl.logging.info(f"Model saved to {model_path}")  
    

  #model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
  #model.save(fn_args.serving_model_dir+'.keras')

# def run_fn(fn_args: TrainerFnArgs):
#     """Train the model based on given args.
#     Args:
#       fn_args: Holds args used to train the model as name/value pairs.
#     """
#     tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)
#     train_dataset = input_fn(
#         fn_args.train_files, 
#         fn_args.data_accessor, 
#         tf_transform_output, 
#         TRAIN_BATCH_SIZE)
#     eval_dataset = input_fn(
#         fn_args.eval_files, 
#         fn_args.data_accessor,
#         tf_transform_output, 
#         EVAL_BATCH_SIZE)
    
#     if fn_args.hyperparameters:
#         hparams = kerastuner.HyperParameters.from_config(fn_args.hyperparameters)
#     else:
#         hparams = get_hyperparameters()
#     absl.logging.info('HyperParameters for training: %s' % hparams.get_config())
    
#     # Distribute training over multiple replicas on the same machine.
#     mirrored_strategy = tf.distribute.MirroredStrategy()
#     with mirrored_strategy.scope():
#         model = build_keras_model(
#             hparams=hparams,
#             tf_transform_output=tf_transform_output)
            
#     tensorboard_callback = tf.keras.callbacks.TensorBoard(
#         log_dir=fn_args.model_run_dir, update_freq='batch')
      
#     model.fit(
#         train_dataset,
#         epochs=EPOCHS,
#         steps_per_epoch=fn_args.train_steps,
#         validation_data=eval_dataset,
#         validation_steps=fn_args.eval_steps,
#         callbacks=[tensorboard_callback])
    
#     # Define a preprocessing function that handles all feature transformations
#     def _preprocessing_fn(features):
#         """Preprocesses features for model prediction."""
#         transformed_features = {}
        
#         # Process numeric features
#         for key in features.NUMERIC_FEATURE_KEYS:
#             transformed_name = features.transformed_name(key)
#             # Handle potential sparse tensor
#             feature = features[key]
#             if isinstance(feature, tf.sparse.SparseTensor):
#                 feature = tf.sparse.to_dense(feature)
#             transformed_features[transformed_name] = tf.cast(feature, tf.float32)
#             if transformed_features[transformed_name].shape.rank == 1:
#                 transformed_features[transformed_name] = tf.expand_dims(transformed_features[transformed_name], -1)
        
#         # Process categorical features 
#         for key in features.CATEGORICAL_FEATURE_KEYS:
#             transformed_name = features.transformed_name(key)
#             # Handle potential sparse tensor
#             feature = features[key]
#             if isinstance(feature, tf.sparse.SparseTensor):
#                 feature = tf.sparse.to_dense(feature)
#             # Use vocabulary mapping for categorical features
#             vocab_file = tf_transform_output.vocabulary_file_by_name(key)
#             vocab_layer = VocabLookupLayer(vocab_file)
#             transformed_features[transformed_name] = vocab_layer(feature)
#             if transformed_features[transformed_name].shape.rank == 1:
#                 transformed_features[transformed_name] = tf.expand_dims(transformed_features[transformed_name], -1)
        
#         return transformed_features
    
#     # Create a serving function that's compatible with TFMA
#     @tf.function
#     def _serving_fn(serialized_tf_examples):
#         """Returns the output to be used in the serving signature."""
#         # Get raw feature spec (excluding label)
#         raw_feature_spec = tf_transform_output.raw_feature_spec().copy()
#         if features.LABEL_KEY in raw_feature_spec:
#             raw_feature_spec.pop(features.LABEL_KEY)
        
#         # Parse the input examples - handle both batch and single inputs
#         parsed_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)
        
#         # Apply preprocessing to the features
#         transformed_features = _preprocessing_fn(parsed_features)
        
#         # Get predictions
#         outputs = model(transformed_features)
        
#         # Format for TFMA - ensure this matches what the evaluator expects
#         return {'outputs': outputs}
    
#     # Create concrete serving function that explicitly handles the input shape
#     concrete_serving_fn = _serving_fn.get_concrete_function(
#         tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
#     )
    
#     # Save the model with appropriate signatures
#     model_dir = fn_args.serving_model_dir
#     signatures = {
#         'serving_default': concrete_serving_fn
#     }
    
#     # Save the model with tensorflow's SavedModel format
#     tf.saved_model.save(
#         model, 
#         model_dir,
#         signatures=signatures
#     )
    
#     # Print confirmation
    # absl.logging.info(f"Model saved to {model_dir} with serving_default signature")
    # Then create a combined model that includes preprocessing and prediction
#   preprocessing_model = _build_preprocessing_model(tf_transform_output)
#   combined_model = _build_combined_model(
#         preprocessing_model, model, tf_transform_output)

#     # Save the combined model with proper signatures
#   combined_model.save(fn_args.serving_model_dir, save_format='tf')

#   absl.logging.info(f"Model saved to {fn_args.serving_model_dir}")
    
  
#   # Create a module for serving with proper signatures
#   class TFXServingModel(tf.Module):
#     def __init__(self, model, transform_output):
#       self.model = model
#       self.transform_output = transform_output
#       self._preprocess_fn = transform_output.transform_features_layer()
    
#     @tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')])
#     def serving_default(self, serialized_tf_examples):
#         """Receives serialized tf.Examples and applies prediction."""
#         # Get raw feature spec for parsing examples
#         raw_feature_spec = self.transform_output.raw_feature_spec()
#         # Remove label feature since it's not available during serving
#         if features.LABEL_KEY in raw_feature_spec:
#             raw_feature_spec.pop(features.LABEL_KEY)

#         # Parse the examples
#         parsed_features = tf.io.parse_example(serialized_tf_examples, raw_feature_spec)

#         # Apply transformations
#         transformed_features = self._preprocess_fn(parsed_features)

#         # Important change here - match the input structure expected by your model
#         # Instead of passing the dictionary directly, pass each input separately matching model's input layer names
#         model_inputs = {}
#         for key in transformed_features:
#             if key in self.model.input_names:
#                 model_inputs[key] = transformed_features[key]

#         # Get predictions by passing the properly structured inputs
#         outputs = self.model(model_inputs)

#         return {'outputs': outputs}
#   # Create the module with serving signature
#   serving_model = TFXServingModel(model, tf_transform_output)
  
#   # Save the model with the required signature
#   serving_model_dir = fn_args.serving_model_dir
#   tf.saved_model.save(
#       serving_model,
#       serving_model_dir,
#       signatures={
#           'serving_default': serving_model.serving_default
#       }
#   )
  
#   absl.logging.info(f"Model saved to {serving_model_dir} with serving_default signature")
     
   



# def _build_preprocessing_model(tf_transform_output):
#     """Creates a TF.Keras model that applies preprocessing transformations."""
#     inputs = {}
    
#     # Create input layers for all raw features
#     for key in features.NUMERIC_FEATURE_KEYS:
#         inputs[key] = tf.keras.layers.Input(
#             shape=(1,), dtype=tf.float32, name=key)
    
#     for key in features.CATEGORICAL_FEATURE_KEYS:
#         inputs[key] = tf.keras.layers.Input(
#             shape=(1,), dtype=tf.string, name=key)
    
#     # Get the transformation layer
#     transform_layer = tf_transform_output.transform_features_layer()
    
#     # Apply the transformation layer to the inputs
#     transformed_features = transform_layer(inputs)
    
#     # Create and return the preprocessing model
#     preprocessing_model = tf.keras.Model(inputs=inputs, outputs=transformed_features)
#     return preprocessing_model

# def _build_combined_model(preprocessing_model, prediction_model, tf_transform_output):
#     """Builds a combined model with preprocessing and prediction."""
#     # Get the input layers from the preprocessing model
#     inputs = preprocessing_model.inputs
    
#     # Get the transformed features from the preprocessing model
#     transformed_features = preprocessing_model(inputs)
    
#     # Filter transformed features to only those expected by the prediction model
#     model_inputs = {}
#     for name in prediction_model.input_names:
#         if name in transformed_features:
#             model_inputs[name] = transformed_features[name]
    
#     # Apply the prediction model to the transformed features
#     outputs = prediction_model(model_inputs)
    
#     # Create and return the combined model
#     combined_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
#     # Compile the model with the same configuration as the prediction model
#     combined_model.compile(
#         loss=prediction_model.loss,
#         optimizer=prediction_model.optimizer,
#         metrics=prediction_model.metrics
#     )
    
#     return combined_model
class VocabLookupLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_file, default_value=-1, **kwargs):
        super(VocabLookupLayer, self).__init__(**kwargs)
        self.vocab_file = vocab_file
        self.default_value = default_value
        self.vocab_table = None

    def build(self, input_shape):
        self.vocab_table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.TextFileInitializer(
                filename=self.vocab_file,
                key_dtype=tf.string,
                key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
                value_dtype=tf.int64,
                value_index=tf.lookup.TextFileIndex.LINE_NUMBER),
            default_value=self.default_value)
        super(VocabLookupLayer, self).build(input_shape)

    def call(self, inputs):
        return self.vocab_table.lookup(inputs)
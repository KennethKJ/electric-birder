import tensorflow.keras.estimator as kes
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout, Flatten, Dense
from train_model.input_fn import input_fn
import tensorflow as tf
from train_model import data_retriaval as dr
import os


def create_estimator(params):

    # learning_rate = params['learning_rate']
    # dropout_rate = params['dropout_rate']
    # num_classes = params['num_classes']

    # Import VGG16 model for transfer learning
    base_model = VGG16(weights='imagenet')
    base_model.summary()

    x = base_model.get_layer('fc2').output

    x = Dropout(params['dropout rate'])(x)

    predictions = Dense(params['num classes'], activation="sigmoid", name="sm_out")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers:
        layer.trainable = False

    for layer in model.layers[20: 23]:
        layer.trainable = True

    model.summary()

    model.compile(
        loss="binary_crossentropy",
        optimizer=tf.train.AdamOptimizer(params['learning rate'],
                                         beta1=0.9,
                                         beta2=0.999),
        metrics=["accuracy"]
    )


    # Set up training config according to Intel recommendations
    NUM_PARALLEL_EXEC_UNITS = 4
    session_config = tf.ConfigProto(
        intra_op_parallelism_threads=NUM_PARALLEL_EXEC_UNITS,
        inter_op_parallelism_threads=2,
        allow_soft_placement=True,
        device_count={'CPU': NUM_PARALLEL_EXEC_UNITS}
    )

    os.environ["OMP_NUM_THREADS"] = "4"
    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"

    run_config = tf.estimator.RunConfig(
        session_config=session_config,
        model_dir=params['output path']
    )

    # Convert to Estimator (https://cloud.google.com/blog/products/gcp/new-in-tensorflow-14-converting-a-keras-model-to-a-tensorflow-estimator)
    estimator_model = kes.model_to_estimator(
        keras_model=model,
        config=run_config
    )

    return estimator_model


# Functions for custom metrics
def precision(labels, predictions):
    prec = tf.metrics.precision(predictions['sm_out'], labels)
    return {'Precision': prec}


def recall(labels, predictions):
    reca = tf.metrics.recall(predictions['sm_out'], labels)
    return {'Recall': reca}


def go_train(params):
    # Create the estimator
    Est = create_estimator(params)

    # Add custom metrics
    Est = tf.contrib.estimator.add_metrics(Est, precision)
    Est = tf.contrib.estimator.add_metrics(Est, recall)

    # Get the data (in form of dictionary)
    data = dr.get_data(params)

    # Set up Estimator train and evaluation specifications
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(data['train_images'], data['train_labels'], params, True)
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(data['eval_images'], data['eval_labels'], params, False),
        steps=10  # Evaluates on 10 batches
    )

    # Set logging level
    tf.logging.set_verbosity(tf.logging.DEBUG)

    # Run training and evaluation
    tf.estimator.train_and_evaluate(Est, train_spec, eval_spec)

# THING TO CHECK OUT

# Add precision, recall etc to summary stuff
# https://stackoverflow.com/questions/49619995/how-to-control-when-to-compute-evaluation-vs-training-using-the-estimator-api-of
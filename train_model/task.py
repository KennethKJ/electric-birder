import argparse
import json
import os

from train_model import model

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath',
        help='GCS path to data. We assume that data is in gs://BUCKET/babyweight/preproc/',
        required=False,
        default='C:/Users/Kenneth Kragh Jensen/Google Drive/ML/Databases/Unified Feeder Birds Database'
    )
    parser.add_argument(
        '--output_dir',
        help='GCS location to write checkpoints and export models',
        required=False,
        default='C:/Users/Kenneth Kragh Jensen/Google Drive/ML/EB/Estimator output/1/'
    )
    parser.add_argument(
        '--batch_size',
        help='Number of examples to compute gradient over.',
        type=int,
        default=26
    )
    parser.add_argument(
        '--job-dir',
        help='this model ignores this field, but it is required by gcloud',
        default='junk'
    )
    parser.add_argument(
        '--numclasses',
        help='Hidden layer sizes to use for DNN feature columns -- provide space-separated layers',
        nargs='+',
        type=int,
        default=10
    )
    parser.add_argument(
        '--train_examples',
        help='Number of examples (in thousands) to run the training job over. If this is more than actual # of examples available, it cycles through them. So specifying 1000 here when you have only 100k examples makes this 10 epochs.',
        type=int,
        default=5000
    )
    parser.add_argument(
        '--eval_steps',
        help='Positive number of steps for which to evaluate model. Default to None, which means to evaluate until input_fn raises an end-of-input exception',
        type=int,
        default=None
    )

    ## parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    # output_dir = os.path.join(
    #     output_dir,
    #     json.loads(
    #         os.environ.get('TF_CONFIG', '{}')
    #     ).get('task', {}).get('trial', '')
    # )

    # Throw properties into params dict to pass to other functions
    params = {}
    params['output path'] = arguments.pop('output_dir')
    params['data path'] = arguments.pop('datapath')
    params['image size'] = [244, 224]
    params['num parallel calls'] = 4
    params["batch size"] = arguments.pop('batch_size')
    params['use random flip'] = True
    params['learning rate'] = 0.007
    params['dropout rate'] = 0.5
    params['num classes'] = 10
    params['train steps'] = (arguments.pop('train_examples') * 1000) / params["batch size"]
    params['eval steps'] = arguments.pop('eval_steps')


    print("Will train for {} steps using batch_size={}".format(params['train steps'], params['batch size']))


    # Run the training job
    model.go_train(params)

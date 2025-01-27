"""
    Top-level model building script using independent modules.
"""
import argparse
import os
import random

import torch

import numpy as np

from validate import exceptions
from validate import validate_params
from data_loaders import data_utils
from abstention import abstention
from models import mthisan, mtcnn
from predict import predictions

# needed for running on the vms
# torch.multiprocessing.set_sharing_strategy('file_system')


def load_model_dict(model_path, data_path):
    """Load pretrained model from disk.

        Args:
            model_path: str, from command line args, points to saved model
            data_path: str, data to make predictions on

        We check if the supplied path is valid and if the packages match needed
            to run the pretrained model.

    """
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}")
        model_dict = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        raise exceptions.ParamError(f'the model at {model_path} does not exist.')

    if not os.path.exists(data_path):
        raise exceptions.ParamError(f'the data at {data_path} does not exist.')

    return model_dict


def load_model(model_dict, device, dw):
    """Load pretrained model architecture."""
    model_args = model_dict['metadata_package']['mod_args']

    if model_args['model_type'] == 'mthisan':
        model = mthisan.MTHiSAN(dw.inference_data['word_embedding'],
                                dw.num_classes,
                                **model_args['MTHiSAN_kwargs'])

    elif model_args['model_type'] == 'mtcnn':
        model = mtcnn.MTCNN(dw.inference_data['word_embedding'],
                            dw.num_classes,
                            **model_args['MTCNN_kwargs'])

    model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model_state_dict = model_dict['model_state_dict']
    model_state_dict = {k.replace('module.',''): v for k, v in model_state_dict.items()
                        if k !=' metadata_package'}
    model.load_state_dict(model_state_dict)

    print('model loaded')

    return model


def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model_path', '-mp', type=str, default='',
                        help="""this is the location of the model
                                that will used to make predictions""")
    parser.add_argument('--data_path', '-dp', type=str, default='',
                        help="""where the data will load from, this must be specified.""")
    parser.add_argument('--model_args', '-args', type=str, default='',
                        help="""path to specify the model_args; default is in
                                the model_suite directory""")

    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    args = parser.parse_args()

    if len(args.model_path) == 0 or len(args.data_path) == 0:
        raise exceptions.ParamError("Model and/or data path cannot be empty, please specify both.")

    # 1. validate model/data args
    print("Validating kwargs in model_args.yml file")
    cache_class = [False]
    data_source = 'pre-generated'
    # use the model args file from training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = load_model_dict(args.model_path, args.data_path)
    mod_args = model_dict['metadata_package']['mod_args']

    print("Validating kwargs from pretrained model ")
    model_args = validate_params.ValidateParams(cache_class, args,
                                                data_source=data_source,
                                                model_args=mod_args)

    model_args.check_data_train_args(from_pretrained=True)
    if model_args.model_args['model_type'] == 'mthisan':
        model_args.hisan_arg_check()
    elif model_args.model_args['model_type'] == 'mtcnn':
        model_args.mtcnn_arg_check()

    if model_args.model_args['abstain_kwargs']['abstain_flag']:
        model_args.check_abstain_args()

    model_args.check_data_files()

    if model_args.model_args['data_kwargs']['reproducible']:
        seed = model_args.model_args['data_kwargs']['random_seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    else:
        seed = None

    # 2. load data
    print("Loading data and creating DataLoaders")

    dw = data_utils.DataHandler(data_source, model_args.model_args, cache_class)
    dw.load_folds(fold=0)

    data_loaders = dw.inference_loader(reproducible=model_args.model_args['data_kwargs']['reproducible'],
                                       seed=seed,
                                       batch_size=model_args.model_args['train_kwargs']['batch_per_gpu'])

    if model_args.model_args['abstain_kwargs']['abstain_flag']:
        dac = abstention.AbstainingClassifier(model_args.model_args, device)
        # modifies n_classes and id2label dict for abstention/ntask
        dac.add_abstention_classes(dw)
        print('Running with:')
        for key in model_args.model_args['abstain_kwargs']:
            print(key, ": ", model_args.model_args['abstain_kwargs'][key])
    else:
        dac = None

    model = load_model(model_dict, device, dw)

    # 5. score predictions from pretrained model or model just trained
    evaluator = predictions.ScoreModel(model_args.model_args, data_loaders, dw, model, device)

    # 5a. score a model
    # evaluator.score(dac=dac)

    # 5b. make predictions
    # evaluator.predict(dw.metadata['metadata'], dw.dict_maps['id2label'])

    # 5c. score and predict
    evaluator.evaluate_model(dw.metadata['metadata'], dw.dict_maps['id2label'], dac=dac)


if __name__ == "__main__":
    main()

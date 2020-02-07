import argparse

TRAIN_ARGS_LIST = {
    'data_dir': {
        'help': 'Directory where dataset is located'
        },
    '--save_dir': {
        'help': 'Directory to save checkpoints',
        'default': None
        },
    '--load_model': {
        'help': 'Full path to a checkpoint, so you can train an existing model',
        'default': None
        },
    '--arch': {
        'help': 'Architecture of the network',
        'choices': ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'alexnet', 'vgg11', 'vgg13', 'vgg19', 'vgg16', 'densenet121', 'densenet161', 'densenet169', 'densenet201'],
        'default': 'densenet121'
        },
    '--learn_rate': {
        'help': 'Learning rate of the network',
        'type': float,
        'default': 0.0009
        },
    '--epochs': {
        'help': 'Number of epochs',
        'type': int,
        'default': 25
        },
    '--input_units': {
        'help': 'Number of units in input layer',
        'type': int,
        'default': 1024
        },
    '--output_units': {
        'help': 'Number of units in output layer',
        'type': int,
        'default': 102
        },
    '--hidden_units': {
        'help': 'Number of neurons in hidden layers',
        'type': int,
        'nargs': '+',
        'default': [512, 256, 128]
        },
    '--print_every': {
        'help': 'Number telling in which steps to print training loss, testing loss and testing accuracy',
        'type': int,
        'default': 10
    },
    '--gpu': {
        'help': 'Whether to use gpu for acceleration or not',
        'dest': 'device',
        'const': 'cuda',
        'default': 'cpu',
        'action': 'store_const'
        },
    '--test': {
        'help': 'After training should we test the network? default is true',
        'type': bool,
        'default': True
        }
    }

PREDICT_ARGS_LIST = {
    'input': {
        'help': 'Full path to the image'
        },
    'checkpoint': {
        'help': 'Path to an existing model\'s checkpoint'
        },
    '--top_k': {
        'help': 'Number of top most likely classes that program will return',
        'type': int,
        'default': 5
        },
    '--category_names': {
        'help': 'Path to a json file containing all the categories',
        'default': './cat_to_name.json'
        },
    '--gpu': {
        'help': 'Whether to use gpu for acceleration or not',
        'dest': 'device',
        'const': 'cuda',
        'default': 'cpu',
        'action': 'store_const'
        }
    }

def create_parser(description, **kargs):
    parser = argparse.ArgumentParser(
        description
        )

    for argument in kargs:
        parser.add_argument(argument, **kargs[argument])

    return parser
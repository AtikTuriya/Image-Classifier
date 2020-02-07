from arguments import TRAIN_ARGS_LIST, create_parser
from utils.network_utils import build_network, train_network, get_loss_function, get_optimizer, test, load_model, save_model
from utils.data_utils import get_dataloaders

def main():
    parser = create_parser(description='Image Classifier', **TRAIN_ARGS_LIST)               # Create an argument parser

    args = parser.parse_args()
    
    print('Loading data..')

    trainloader, validloader, testloader, class_to_idx = get_dataloaders(args.data_dir)     # Get all the dataloaders

    print('Building network..')

    if args.load_model:
        model, network_arch = load_model(args.load_model)                                   # load an existing model
    else:
        network_arch = {
            'architecture': args.arch,
            'input': args.input_units,
            'hidden_layers': args.hidden_units,
            'output': args.output_units
        }
        model = build_network(**network_arch)                                               # build a new model
        model.class_to_idx = class_to_idx
    
    print('Creating optimizer and loss function..')
    optimizer = get_optimizer(model, args.learn_rate)
    criterion = get_loss_function()

    print('Initializing training..')

    train_network(
        model,
        trainloader,
        validloader,
        optimizer,
        criterion,
        args.epochs,
        args.print_every,
        args.device
    )                                                                                       # train the network

    if args.save_dir:                                                                       # Save the network if save_dir is present, with full path
        print('Saving model..')
        save_model(model, network_arch, args.save_dir)
        print('Saved successfully.')

    if args.test:                                                                           # Test the accuracy of the model if test was passed in arguments
        print('Testing the network now..')
        print("Accuracy of the network on test images is :", test(model, testloader, args.device))


if __name__ == '__main__':
    main()
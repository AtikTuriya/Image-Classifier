from torchvision import models
import torch
from torch import nn, optim
from collections import OrderedDict
from .image_utils import process_image

network_names  = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'alexnet', 'vgg11', 'vgg13', 'vgg19', 'vgg16', 'densenet121', 'densenet161', 'densenet169', 'densenet201']
network_models = [
    models.resnet18, models.resnet34, models.resnet50, models.resnet101,
    models.alexnet,
    models.vgg11, models.vgg13, models.vgg19, models.vgg16,
    models.densenet121, models.densenet161, models.densenet169, models.densenet201
]
networks = dict(zip(network_names, network_models))


def build_network(architecture='densenet121', input=1024, hidden_layers=[512, 256, 128], output=102):
    model_to_use = networks.get(architecture, models.densenet121)
    model = model_to_use(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    layers = [('inputs', nn.Linear(input, hidden_layers[0]))]
    i = 1
    #dropouthere = len(layers) // 2
    for inu, outu in zip(hidden_layers[:-1], hidden_layers[1:]):
        layers.append(('relu%d' % i, nn.ReLU()))
        layers.append(('hidden_layers%d' % i, nn.Linear(inu, outu)))
        #if i == dropouthere:
        #    layers.append(('dropout', nn.Dropout(0.1)))
        i += 1

    layers.append(('dropout', nn.Dropout(0.1)))
    layers.append(('hidden_layer%d' % i, nn.Linear(hidden_layers[-1], output)))
    layers.append(('output', nn.LogSoftmax(dim=1)))

    classifier = nn.Sequential(OrderedDict(layers))

    model.classifier = classifier

    return model

def save_model(model, network_arch, fullpath='./model.pth'):
    """network_arch is a dictionary that will hold information to recreate the entire network whenever we are reloading our model.
    So network_arch will have such structure,
        {
            architecture: name of the network architecture, e.g. densenet121,
            input: no. of input units
            hidden_layers: [a list of integers giving no. of units in each hidden layer], e.g. [512, 256, 128],
            output: no. of output units
        }
    """
    checkpoint = {
        'architecture' : network_arch,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict()
    }

    torch.save(checkpoint, fullpath)

def load_model(path):
    checkpoint = torch.load(path)
    model = build_network(**checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model, checkpoint['architecture']

def get_optimizer(model, learn_rate):
    return torch.optim.Adam(model.classifier.parameters(), learn_rate)

def get_loss_function():
    return nn.NLLLoss()

def train_network(model, trainloader, validloader, optimizer, criterion, epochs=25, steps=10, device='cpu'):
    if device == 'cuda':
        model.cuda()
    for e in range(epochs):
        running_loss = 0
        for ix, (image, label) in enumerate(trainloader):
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()

            output = model(image)
            loss   = criterion(output, label)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if ix % steps == 0: 
                test_loss, accuracy = validate(model, validloader, criterion, device)
                print(
                    f"Epoch {e+1}/{epochs}.. "
                    f"Train loss: {running_loss/steps:.3f}.. "
                    f"Test loss: {test_loss/len(validloader):.3f}.. "
                    f"Test accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0

def validate(model, dataloader, criterion, device='cpu'):
    test_loss = 0
    accuracy  = 0
    model.eval()
    with torch.no_grad():
        for input, label in dataloader:
            input, label = input.to(device), label.to(device)
            logps = model.forward(input)
            batch_loss = criterion(logps, label)

            test_loss += batch_loss.item()

            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == label.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    model.train()
    return test_loss, accuracy

def test(model, dataloader, device='cpu'):
    correct = 0
    total   = 0
    model.eval()
    model.to(device)
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            output = model(image)
            _, predicted = torch.max(output.data, 1)
            total   += label.size(0)
            correct += (predicted == label).sum().item()

    return 100 * correct / total

def predict(model, image_path, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    if device == 'cuda':
        model.cuda()
    
    model.eval()
    img = process_image(image_path)
    img = torch.from_numpy(img).float()

    if device == 'cuda':
        img = img.cuda()
    
    img = torch.unsqueeze(img, dim=0)

    output  = model.forward(img)
    preds   = torch.exp(output).topk(topk)
    probs   = preds[0][0].cpu().data.numpy()
    classes = preds[1][0].cpu().data.numpy()

    topk_labels = [idx_to_class[i] for i in classes]

    return probs.tolist(), topk_labels
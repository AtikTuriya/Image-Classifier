from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import json

MEAN     = [0.485, 0.456, 0.406]
STD      = [0.229, 0.224, 0.225]
DATA_DIR = './flowers'

TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

VALID_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

TEST_TRANSFORM  = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def get_dataloaders(data_dir=DATA_DIR):
    """
    """
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'

    train_data = datasets.ImageFolder(train_dir, transform=TRAIN_TRANSFORM)
    valid_data = datasets.ImageFolder(valid_dir, transform=VALID_TRANSFORM)
    test_data  = datasets.ImageFolder(test_dir,  transform=TEST_TRANSFORM)

    trainloader = DataLoader(train_data, batch_size=50, shuffle=True)
    validloader = DataLoader(valid_data, batch_size=50)
    testloader  = DataLoader(test_data,  batch_size=50)

    return trainloader, validloader, testloader, train_data.class_to_idx

def load_categories(path_to_json):
    return json.load(open(path_to_json, 'r'))
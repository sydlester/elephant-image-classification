import pandas as pd
from torch.utils.data import DataLoader, random_split
import torch
import os
from torchvision import transforms
from data.ImageDataset import CustomImageDataset
from model.ResNet import ResNet
import argparse

def get_data(input_data):
    # read in the data
    consensus_data = pd.read_csv('data/consensus_data.csv')[['CaptureEventID', 'Species']]
    images = pd.read_csv('data/all_images.csv')

    # create a dataframe with the image urls and species label
    df = pd.merge(images, consensus_data, on='CaptureEventID')
    df["URL"] = df["URL_Info"]
    df = df[['URL', 'Species']]

    # define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # create the train & test dataloaders
    train_data = CustomImageDataset(df=df, root_dir=input_data, transform=transform)
    train_set, test_set = random_split(train_data, [0.7, 0.3])
    # update num_workers?
    train_dataloader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_set, batch_size=64, shuffle=True, num_workers=4)

    return train_dataloader, test_dataloader

def train_model(train_dataloader, test_dataloader, model, criterion, optimizer, scheduler, num_epochs=5):
    num_batches = len(train_dataloader)
    print(f'Number of batches: {num_batches}')

    classes = ['human', 'gazelleGrants', 'reedbuck', 'dikDik', 'zebra', 'porcupine',
                'gazelleThomsons', 'hyenaSpotted', 'warthog', 'impala', 'elephant', 'giraffe',
                'mongoose', 'buffalo', 'hartebeest', 'guineaFowl', 'wildebeest', 'leopard',
                'ostrich', 'lionFemale', 'koriBustard', 'otherBird', 'batEaredFox', 'bushbuck',
                'jackal', 'cheetah', 'eland', 'aardwolf', 'hippopotamus', 'hyenaStriped',
                'aardvark', 'hare', 'baboon', 'vervetMonkey', 'waterbuck', 'secretaryBird',
                'serval', 'lionMale', 'topi', 'honeyBadger', 'rodents', 'wildcat', 'civet',
                'genet', 'caracal', 'rhinoceros', 'reptiles', 'zorilla'
    ]
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if (phase == 'train'):
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                dataloader = train_dataloader

            running_loss = 0.0
            running_corrects = 0
            i = 0
            for inputs, labels in dataloader:
                inputs = torch.tensor(inputs)
                labels = torch.tensor([classes.index(label) for label in labels])
                # inputs, labels = inputs.to(gpu), labels.to(gpu)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                loss = loss.item() * inputs.size(0)
                corrects = torch.sum(preds == labels.data)
                if i % 100 == 0:
                    print(f'Batch {i} Loss: {loss:.4f}, Correct: {corrects.item()}')

                i += 1
                running_loss += loss
                running_corrects += corrects

            if (phase == 'train'):
                scheduler.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Accuracy: {epoch_acc:.4f}')

    return model

def main(input_data):
    input_data_path = input_data
    print(f"Input data path: {input_data_path}")
    # get the dataloaders
    train_dataloader, test_dataloader = get_data(input_data_path)
    # load the model
    model = ResNet(num_classes=48)

    # Define the loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # Train the model
    epochs = 10
    model = train_model(train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, num_epochs=epochs)

    output_dir = "./outputs"  # Azure ML automatically tracks this directory
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "elephant_classifier_resnet50.pth")

    torch.save(model.state_dict(), model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, help="Path to the input data")
    args = parser.parse_args()

    main(args.input_data)
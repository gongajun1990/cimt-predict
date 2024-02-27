import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import  DataLoader
from ModelAndEyeDataset import SiameseResNeXtWithAge,EyeDataset,SiameseResNeXtWithAge2
from utilstools.utils import ValidTransform, TrainTransform, calculate_weights

# Detect GPU and select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=25, log_file='train_log.txt'):
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for left_eye, right_eye, labels, age in train_loader:
            left_eye = left_eye.to(device)
            right_eye = right_eye.to(device)
            labels = labels.to(device)
            age = age.to(device)
            optimizer.zero_grad()

            # Note that the call to this model already includes the age feature
            outputs = model(left_eye, right_eye,age)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * left_eye.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        valid_loss, valid_acc = validate_model(model, valid_loader, criterion)

        # Check if it is the best model
        if valid_acc > best_acc and epoch_acc > valid_acc:
            best_acc = valid_acc
            filename = f"{modleName}model_state_dict_{best_acc:.4f}.pth"
            torch.save(model.state_dict(), filename)  # save best model

        with open(log_file, 'a') as f:
            f.write(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}\n')
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}\n')

        scheduler.step()
    return model

# valid function
def validate_model(model, valid_loader, criterion):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for left_eye, right_eye, labels, age in valid_loader:
            left_eye = left_eye.to(device)
            right_eye = right_eye.to(device)
            labels = labels.to(device)
            age = age.to(device)
            outputs = model(left_eye, right_eye, age)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * left_eye.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
    total_loss = running_loss / len(valid_loader.dataset)
    total_acc = running_corrects.double() / len(valid_loader.dataset)
    return total_loss, total_acc


def main(TRAIN_DATA_PATH, VALID_DATA_PATH,weight_path,model,num_epochs=200,lr=0.001, age_path='age.xls'):
    train_dataset = EyeDataset(TRAIN_DATA_PATH, transform=TrainTransform, age_path=age_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, prefetch_factor=2)
    valid_dataset = EyeDataset(VALID_DATA_PATH, transform=ValidTransform, age_path=age_path)
    valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=4, prefetch_factor=2)
    model=model

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    weights = calculate_weights(TRAIN_DATA_PATH)  #
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if weight_path and os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path))


    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler,num_epochs=num_epochs)


if __name__ == '__main__':
    # global parameters
    TRAIN_DATA_PATH = r'H:\eye-cmit\train'  # training data path
    VALID_DATA_PATH = r'H:\eye-cmit\valid'  # valid  data path
    weight_path = r'SiameseResNeXtWithAgemodel_state_dict_0.8300.pth'
    age_path = './age.xls'

    model = SiameseResNeXtWithAge().to(device)

    modleName=model.__class__.__name__


    #train epochs
    num_epochs=100
    #Learning rate parameters
    lr=0.001
    main(TRAIN_DATA_PATH=TRAIN_DATA_PATH, VALID_DATA_PATH=VALID_DATA_PATH, weight_path=weight_path, model=model,
         num_epochs=num_epochs, lr=lr, age_path=age_path)


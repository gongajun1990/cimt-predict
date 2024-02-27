import os
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from ModelAndEyeDataset import SiameseResNet,SiameseResNeXt,EyeDataset
from utilstools.utils import ValidTransform, TrainTransform, calculate_weights

# Detect GPU and select device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#Training model function
def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler, num_epochs=25):
    best_acc = 0.0  #Initialize the best accuracy

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        # Training process
        for i, (left_eye, right_eye, labels) in enumerate(train_loader):
            left_eye = left_eye.to(device)
            right_eye = right_eye.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(left_eye, right_eye)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * left_eye.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # # Valid process
        valid_loss, valid_acc = validate_model(model, valid_loader, criterion, epoch)

        # Check if it is the best model
        if valid_acc > best_acc and epoch_acc > valid_acc:
            best_acc = valid_acc
            filename = f"{modleName}model_state_dict_{best_acc:.4f}.pth"

            torch.save(model.state_dict(), filename)  # Save the best model

        print(
            f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}\n')

        scheduler.step()


# valid funcion
def validate_model(model, valid_loader, criterion, epoch, log_file=r'./valid_log.txt'):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for i, (left_eye, right_eye, labels) in enumerate(valid_loader):
            left_eye = left_eye.to(device)
            right_eye = right_eye.to(device)
            labels = labels.to(device)

            outputs = model(left_eye, right_eye)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * left_eye.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(valid_loader.dataset)
    total_acc = running_corrects.double() / len(valid_loader.dataset)

    # 可选：将验证信息写入日志
    with open(log_file, 'a') as f:
        f.write(f'Epoch {epoch + 1}, Validation Loss: {total_loss:.4f}, Accuracy: {total_acc:.4f}\n')

    return total_loss, total_acc

def main(TRAIN_DATA_PATH, VALID_DATA_PATH,weight_path,model,num_epochs=200,lr=0.001):
    train_dataset = EyeDataset(TRAIN_DATA_PATH, transform=TrainTransform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8, prefetch_factor=2)
    valid_dataset = EyeDataset(VALID_DATA_PATH, transform=ValidTransform)
    valid_loader = DataLoader(valid_dataset, batch_size=32, num_workers=4, prefetch_factor=2)
    model=model

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    weights = calculate_weights(TRAIN_DATA_PATH)  #
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if weight_path and os.path.isfile(weight_path):
        model.load_state_dict(torch.load(weight_path))

    train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler,num_epochs=num_epochs)


if __name__ == '__main__':
    # 全局参数
    TRAIN_DATA_PATH = r'H:\eye-cmit\train'  # training data path
    VALID_DATA_PATH = r'H:\eye-cmit\valid'  # valid  data path
    weight_path = r''

    model = SiameseResNeXt().to(device)
    modleName=model.__class__.__name__

      # training epoch
    num_epochs = 100
    # Learning rate parameter
    lr = 0.001

    main(TRAIN_DATA_PATH=TRAIN_DATA_PATH, VALID_DATA_PATH=VALID_DATA_PATH, weight_path=weight_path,model=model,num_epochs=num_epochs,lr=lr)

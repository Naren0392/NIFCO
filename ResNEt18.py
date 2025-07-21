# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, models, transforms
# from torch.utils.data import DataLoader
# import os

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print("Using device:", device)

#     train_dir = r"C:\Users\Admin\Downloads\ast\vIT\train - Copy"
#     val_dir = r"C:\Users\Admin\Downloads\ast\vIT\valid - Copy"

#     transform = {
#         'train': transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5]*3, [0.5]*3)
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5]*3, [0.5]*3)
#         ])
#     }

#     train_dataset = datasets.ImageFolder(train_dir, transform=transform['train'])
#     val_dataset = datasets.ImageFolder(val_dir, transform=transform['val'])

#     train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
#     val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

#     num_classes = len(train_dataset.classes)
#     print("Classes:", train_dataset.classes)

#     # ✅ Load ResNet18
#     from torchvision.models import resnet18, ResNet18_Weights
#     model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

#     # ✅ Replace the fully connected (classifier) layer
#     model.fc = nn.Linear(model.fc.in_features, num_classes)
#     model = model.to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.0001)

#     num_epochs = 60
#     for epoch in range(num_epochs):
#         print(f"\nEpoch {epoch+1}/{num_epochs}")
#         print("-" * 30)

#         for phase in ['train', 'val']:
#             model.train() if phase == 'train' else model.eval()
#             dataloader = train_loader if phase == 'train' else val_loader
#             dataset_size = len(train_dataset if phase == 'train' else val_dataset)

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in dataloader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / dataset_size
#             epoch_acc = running_corrects.double() / dataset_size

#             print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

#     print("Training completed.")
#     torch.save(model.state_dict(), "resnet18_lhs_rhs_classifier23435&36.pth")

# if __name__ == '__main__':
#     main()
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ✅ Change train and validation directory paths accordingly
    train_dir = r"D:\downloads_archieve_14_07_2025\New folder\train1"
    val_dir = r"D:\downloads_archieve_14_07_2025\New folder\valid"

    transform = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
    }

    train_dataset = datasets.ImageFolder(train_dir, transform=transform['train'])
    val_dataset = datasets.ImageFolder(val_dir, transform=transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    num_classes = len(train_dataset.classes)
    print("Classes:", train_dataset.classes)  # ✅ Should show ['26116_RH', '26117_LH', 'New_part']

    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # ✅ Update final classification layer for 3 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 40
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            dataloader = train_loader if phase == 'train' else val_loader
            dataset_size = len(train_dataset if phase == 'train' else val_dataset)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print("Training completed.")
    torch.save(model.state_dict(), "resnet18_rhs_lhs_newpart9.pth")


if __name__ == '__main__':
    main()

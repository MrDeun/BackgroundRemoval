import torch
import torch.nn as nn
import torchvision
from mpmath.identification import transforms
from torch.utils.data import DataLoader
from CocoDataset import CocoSegmentationDataset
from AutoEncoder import AutoEncoder
import torchvision.transforms as transforms_
def train_autoencoder(model, dataloader, epochs=10, lr=0.001):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    criterion = nn.BCELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    max_step = 200

    for epoch in range(epochs):
        model.train()
        step = 0
        epoch_loss = 0
        for images, masks in dataloader:
            if step > max_step:
                break
            masks = masks/masks.max()
            # print(masks.unique())
            images, masks = images.to(device), masks.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(dataloader):.4f}")

    print("Training complete!")
    return model

base_path = '.'

if __name__ == '__main__':
    dataset = CocoSegmentationDataset(
        root='{}/CocoDatasets/coco2017/train2017'.format(base_path),
        annFile='{}/CocoDatasets/coco2017/annotations/instances_train2017.json'.format(base_path),
        target_size=(256,256)
    )

    dataloader = DataLoader(dataset,batch_size=16,shuffle=True)

    model = AutoEncoder()
    trained_model = train_autoencoder(model,dataloader=dataloader,epochs=5)
    torch.save(model.state_dict(),'{}/out/background_model.pth'.format(base_path))
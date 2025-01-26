import torch
import torchsummary
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from torchvision.io import read_image
import cv2
from AutoEncoder import AutoEncoder
import numpy as np

base_path = '.'

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    weights = torch.load("{}/out/background_model.pth".format(base_path), map_location=device)
    model = AutoEncoder()
    model.load_state_dict(weights)

    model.to(device)
    model.eval()
    torchsummary.summary(model,input_size=(3,256,256))

    img = cv2.imread("{}/proxy-image.jpg".format(base_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb,(256,256))
    img_normalized = img/255.0
    torch_image = torch.from_numpy(img_normalized).float().permute(2,0,1).unsqueeze(0).to(device)
    print(torch_image.shape)
    output = model(torch_image)

    output_image = output.cpu().detach().numpy()[0]
    print(output_image.shape)
    if output_image.shape[0] == 1:  # RGB image (C, H, W)
        output_image = output_image.transpose(1, 2, 0)
    binary_mask = (output_image > 0.3).astype(np.uint8)
    binary_mask *= 255
    binary_mask = cv2.resize(binary_mask,(img.shape[1],img.shape[0]))
    result = cv2.bitwise_and(img, img, mask=binary_mask)
    cv2.imshow("result", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



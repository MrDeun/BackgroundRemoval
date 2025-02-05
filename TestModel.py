import torch
import torchsummary
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import cv2
from AutoEncoder import AutoEncoder
import numpy as np
import os
from os import walk

base_path = '.'

def get_test_images(): 
    filenames = next(walk('{}/test/'.format(base_path)), (None, None, []))[2]  # [] if no file
    return filenames


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    test_images_paths = get_test_images()

    weights = torch.load("{}/out/background_model.pth".format(base_path), map_location=device,weights_only=True)
    model = AutoEncoder()
    model.load_state_dict(weights)

    model.to(device)
    model.eval()
    torchsummary.summary(model,input_size=(3,256,256))


    test_length = len(test_images_paths)
    imgs = []
    masks = []
    filtered_images = []
    for i in range(len(test_images_paths)):
        filename = test_images_paths[i]
        img = cv2.imread("{}/test/{}".format(base_path,filename))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb,(256,256))
        img_normalized = img/255.0
        imgs.append(img_rgb)
        torch_image = torch.from_numpy(img_normalized).float().permute(2,0,1).unsqueeze(0).to(device)
        print(torch_image.shape)
        output = model(torch_image)

        output_image = output.cpu().detach().numpy()[0]
        if output_image.shape[0] == 1:  # RGB image (C, H, W)
            output_image = output_image.transpose(1, 2, 0)
        masks.append(output_image)
        binary_mask = (output_image > 0.3).astype(np.uint8)
        binary_mask *= 255
        binary_mask = cv2.resize(binary_mask,(img.shape[1],img.shape[0]))
        result = cv2.bitwise_and(img, img, mask=binary_mask)
        filtered_images.append(cv2.cvtColor(result,cv2.COLOR_BGR2RGB))

    print(test_length)

    for i in range(len(imgs)):
        plt.subplot(test_length,3,3*i+1)
        plt.imshow(np.array(imgs[i]))
        plt.axis('off')
        plt.subplot(test_length,3,3*i+2)
        plt.imshow(np.array(masks[i]),cmap='gray')
        plt.axis('off')
        plt.subplot(test_length,3,3*i+3)
        plt.imshow(np.array(filtered_images[i]))
        plt.axis('off')

    plt.show()
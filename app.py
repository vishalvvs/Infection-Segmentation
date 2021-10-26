from flask import Flask, jsonify, request, render_template, redirect
import os
import subprocess, base64
import cv2
from PIL import Image
from classification.classification import predict
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn as nn
import torch.nn.functional as F
from ml_backend.bts.model import DynamicUNet
# from ml_backend.bts.classifier import BrainTumorClassifier


app = Flask(__name__)

app.config["IMAGE_UPLOADS"] = "/input_imgs"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]
app.config["IMAGE_HEATMAP"] = "/heat_map"

class DynamicUNet(nn.Module):
    """ 
    The input and output of this network is of the same shape.
    Input Size of Network - (1,512,512)
    Output Size of Network - (1,512,512)
    Shape Format :  (Channel, Width, Height)
    """

    def __init__(self, filters, input_channels=1, output_channels=1):
        """ Constructor for UNet class.
        Parameters:
            filters(list): Five filter values for the network.
            input_channels(int): Input channels for the network. Default: 1
            output_channels(int): Output channels for the final network. Default: 1
        """
        super(DynamicUNet, self).__init__()

        if len(filters) != 5:
            raise Exception(f"Filter list size {len(filters)}, expected 5!")

        padding = 1
        ks = 3
        # Encoding Part of Network.
        #   Block 1
        self.conv1_1 = nn.Conv2d(input_channels, filters[0], kernel_size=ks, padding=padding)
        self.conv1_2 = nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2)
        #   Block 2
        self.conv2_1 = nn.Conv2d(filters[0], filters[1], kernel_size=ks, padding=padding)
        self.conv2_2 = nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2)
        #   Block 3
        self.conv3_1 = nn.Conv2d(filters[1], filters[2], kernel_size=ks, padding=padding)
        self.conv3_2 = nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2)
        #   Block 4
        self.conv4_1 = nn.Conv2d(filters[2], filters[3], kernel_size=ks, padding=padding)
        self.conv4_2 = nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.maxpool4 = nn.MaxPool2d(2)
        
        # Bottleneck Part of Network.
        self.conv5_1 = nn.Conv2d(filters[3], filters[4], kernel_size=ks, padding=padding)
        self.conv5_2 = nn.Conv2d(filters[4], filters[4], kernel_size=ks, padding=padding)
        self.conv5_t = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)

        # Decoding Part of Network.
        #   Block 4
        self.conv6_1 = nn.Conv2d(filters[4], filters[3], kernel_size=ks, padding=padding)
        self.conv6_2 = nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.conv6_t = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        #   Block 3
        self.conv7_1 = nn.Conv2d(filters[3], filters[2], kernel_size=ks, padding=padding)
        self.conv7_2 = nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.conv7_t = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        #   Block 2
        self.conv8_1 = nn.Conv2d(filters[2], filters[1], kernel_size=ks, padding=padding)
        self.conv8_2 = nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.conv8_t = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        #   Block 1
        self.conv9_1 = nn.Conv2d(filters[1], filters[0], kernel_size=ks, padding=padding)
        self.conv9_2 = nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)

        # Output Part of Network.
        self.conv10 = nn.Conv2d(filters[0], output_channels, kernel_size=ks, padding=padding)

    def forward(self, x):
        """ Method for forward propagation in the network.
        Parameters:
            x(torch.Tensor): Input for the network of size (1, 512, 512).

        Returns:
            output(torch.Tensor): Output after the forward propagation 
                                    of network on the input.
        """

        # Encoding Part of Network.
        #   Block 1
        conv1 = F.relu(self.conv1_1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.maxpool1(conv1)
        #   Block 2
        conv2 = F.relu(self.conv2_1(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.maxpool2(conv2)
        #   Block 3
        conv3 = F.relu(self.conv3_1(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.maxpool3(conv3)
        #   Block 4
        conv4 = F.relu(self.conv4_1(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.maxpool4(conv4)

        # Bottleneck Part of Network.
        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))

        # Decoding Part of Network.
        #   Block 4
        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)
        conv6 = F.relu(self.conv6_1(up6))
        conv6 = F.relu(self.conv6_2(conv6))
        #   Block 3
        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)
        conv7 = F.relu(self.conv7_1(up7))
        conv7 = F.relu(self.conv7_2(conv7))
        #   Block 2
        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)
        conv8 = F.relu(self.conv8_1(up8))
        conv8 = F.relu(self.conv8_2(conv8))
        #   Block 1
        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)
        conv9 = F.relu(self.conv9_1(up9))
        conv9 = F.relu(self.conv9_2(conv9))

        # Output Part of Network.
        output = F.sigmoid(self.conv10(conv9))

        return output

def get_model_output(image, model):
  """Returns the saved model output"""
  image = image.view((-1, 1, 512, 512)).to('cpu')
  output = model(image).detach().cpu()
  output = (output > 0.5)
  output = output.numpy()
  output = np.resize((output * 255), (512, 512))
  return output

def get_file(file_name):
  """Load the image by taking file name as input"""
  default_transformation = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((512, 512))
  ])

  image = default_transformation(Image.open(file_name))
  return TF.to_tensor(image)

def image_2_heatmap(img_path, mask_path):
    print("img_path", img_path)
    print("mask_path", mask_path)
    img = cv2.imread(str(img_path), 0)
    img = cv2.resize(img, (208, 176), interpolation = cv2.INTER_CUBIC)
    print("IMAGE DONE")
    heatmap=  cv2.imread(str(mask_path), 0)
    heatmap = cv2.resize(heatmap, (208, 176), interpolation = cv2.INTER_CUBIC)
    img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_PINK)
    image = 0.6*img+0.4*heatmap
    return image


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/<link>')
def notfound(link):
    if(link != 'upload-image'):
        return '404 NOT FOUND'
    else:
        return redirect('/upload-image')

@app.route('/upload-image', methods=['GET', 'POST'])
def upload_img():
    filter_list = [16, 32, 64, 128, 256]
    model = DynamicUNet(filter_list)
    model.load_state_dict(torch.load(r"ml_backend/saved_models/UNet-[16, 32, 64, 128, 256].pt", map_location=torch.device('cpu')))
    if request.method == "POST":
        if request.files:
            image = request.files["image"]
            if(image.filename == ''):
                return "NO FILE UPLOADED"
            filename =image.filename 
            name, ext = filename.split(".")
            
            if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
                # image.save(os.path.join(app.config["IMAGE_UPLOADS"], image.filename))
                print("IMAGE TYPE is: ",type(image))
                dumm = "input_imgs/" + image.filename
                image.save(dumm)
                mask_img = get_file(f'input_imgs/{name}.{ext}')
                op = get_model_output(mask_img,model)
                cv2.imwrite(f'output_imgs/{name}_predicted.{ext}',op)
                # os.chdir('ml_backend')
                # subprocess.Popen(f'python3 api.py --file ../input_imgs/{filename}', shell=True)
                # os.chdir('../') 
                img = image_2_heatmap(f'input_imgs/{name}.{ext}', f'output_imgs/{name}_predicted.{ext}')
                cv2.imwrite(f'heat_map/{name}_map.{ext}',img)
                print(os.getcwd()+f'heatmap/{name}_map.{ext}')
                # cv2.imwrite(os.path.join(app.config["IMAGE_HEATMAP"], f'{name}_map.{ext}'), img)
                with open(f'heat_map/{name}_map.{ext}', 'rb') as out_raw:
                    out_img64 = base64.b64encode(out_raw.read())
                out_img64 = out_img64.decode("utf-8")
                prediction = predict(f'input_imgs/{filename}', 'classification/saved_weight/current_checkpoint.pt')
                print(prediction)
                return render_template('upload_img.htm', predicted = True, imgData = out_img64, supplied_text = f'{prediction}')
            else:
                return "NON SUPPORTED FILE TYPE"
            return redirect(request.url)
    return render_template('upload_img.htm', predicted = False)

if __name__ == '__main__':
    app.run(debug=True)


from flask import Flask, request, jsonify, render_template, url_for, redirect
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
import uuid
from werkzeug.utils import secure_filename

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, datasets
import matplotlib.pyplot as plt

class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.1),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.2),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=5),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.2),

            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128,256, kernel_size=3),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.2),
            nn.Linear(16384, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),

            nn.Dropout(p = 0.2),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),

            nn.Dropout(p = 0.1),
            nn.Linear(128, 6),
#             nn.BatchNorm1d(512), not in this model
            nn.Softmax(dim=1)
        )



    def forward(self, x):
        x = self.features(x)
        x = x.reshape(-1, 16384)
        x = self.classifier(x)

        return x

def get_num_correct(pred, label):
    return pred.argmax(dim=1).eq(label).sum().item()


device = torch.device("cuda:0")


torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)

PATH_model = "modelv5.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net = Network()
net.load_state_dict(torch.load(PATH_model, map_location=device))

print("Model loaded successfully!")

torch.save(net.state_dict(), PATH_model)


print("Good")


from PIL import Image
from torchvision import transforms

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((800, 800)),  # Resize image to the size expected by the model
    transforms.ToTensor(),          # Convert the image to a tensor
])

# Before

# Function to predict the class from an image
# def predict_image(image_path):
#     # Load and preprocess the image
#     image = Image.open(image_path).convert('RGB')
#     image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
#     print("Image" , image)

#     # Put the model in evaluation mode and make predictions
#     net.eval()
#     with torch.no_grad():
#         output = net(image)
    
#     print("Output: " , output)
    
#     # Get the predicted class
#     predicted_class = output.argmax(dim=1)
#     print("class" , predicted_class)
#     return predicted_class

# After

def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    print("Image:", image)

    # Put the model in evaluation mode and make predictions
    net.eval()
    with torch.no_grad():
        output = net(image)
    
    print("Output:", output)

    # Map predicted class to the same label space as in the first code block
    predicted_class = output.argmax(dim=1)  # Get the index of the highest score
    # predicted_class_label = test_label_map_dict[predicted_class.item()]  # Convert index to label if needed

    print("Predicted Class (Index):", predicted_class)
    # print("Predicted Class (Label):", predicted_class_label)

    return predicted_class






# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/home")
def home():
    result = request.args.get("result", None)
    image_url = request.args.get("image_url", None)
    return render_template("upload.html", result=result, image_url=image_url)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        # Save the image temporarily
        filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[-1])
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(file_path)

        # Perform inference using the imported function
        result = predict_image(file_path)
        print(result)
        
        # Redirect to home page with the result
        return redirect(url_for('home', result=result, image_url=url_for('static', filename=f'uploads/{filename}')))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

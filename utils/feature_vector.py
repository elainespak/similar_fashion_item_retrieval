# -*- coding: utf-8 -*-


import re
import glob
import tqdm
import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms


pd.set_option('display.max_columns', 20)
dat = pd.read_csv('mytheresa_all.csv')
ImageFile.LOAD_TRUNCATED_IMAGES = True # prevents bit error


# Load the pretrained model
model = models.resnet50(pretrained=True) # 18, 34, 50, 101, 152
# Use the model object to select the desired layer
layer = model._modules.get('avgpool')

# Set model to evaluation mode
model.eval()

scaler = transforms.Resize((224, 224))
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
to_tensor = transforms.ToTensor()

def get_feature_vector(image_name):
    # https://www.stacc.ee/extract-feature-vector-image-pytorch/
    # 1. Load the image with Pillow library
    img = Image.open(image_name).convert('RGB')
    # 2. Create a PyTorch Variable with the transformed image
    t_img = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    # 3. Create a vector of zeros that will hold our feature vector
    #    Resnet18, 34: The 'avgpool' layer has an output size of 512
    #    Resnet50: The 'avgpool' layer has an output size of 2048
    my_embedding = torch.zeros(2048)
    # 4. Define a function that will copy the output of a layer
    def copy_data(m, i, o):
        #my_embedding.copy_(o.data)
        my_embedding.copy_(o.data.squeeze())
    # 5. Attach that function to our selected layer
    h = layer.register_forward_hook(copy_data)
    # 6. Run the model on our transformed image
    model(t_img)
    # 7. Detach our copy function from the layer
    h.remove()
    # 8. Return the feature vector
    return my_embedding


"""
# lsh similarity --- not as great!
from lshash2 import LSHash

k = 10 # hash size
L = 5  # number of tables
d = 512 # Dimension of Feature vector
lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)
for img_path, vec in tqdm(vecs.items()):
    lsh.index(vec.flatten(), extra_data=img_path)

q_vec = list(vecs.values())[idx]
response = lsh.query(q_vec, num_results= 5)[3]
Image.open(path+response[0][1])
"""
            

if __name__== "__main__":
    
    # Parameters
    files = glob.glob(r'./data/mytheresa/mytheresa_preprocessed/*.jpg')
    path = './data/mytheresa/mytheresa_preprocessed/'
    outfit_path = './data/mytheresa/mytheresa_outfit/'
    model_path = './models/'
    
    # extract feature vectors
    vecs = {}
    for f in tqdm(files, position=0, leave=True):
        vecs[re.search('(?<=\\\).*', f).group(0)] = get_feature_vector(f)
    
    # save
    np.save(model_path+"mytheresa_raw_resnet50_avgpool.npy", vecs, allow_pickle=True)
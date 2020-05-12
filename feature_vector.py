# -*- coding: utf-8 -*-

import pandas as pd
pd.set_option('display.max_columns', 20)
dat = pd.read_csv('mytheresa_all.csv')

#== setup =================================================================
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image, ImageFile
import re
from tqdm import tqdm, tqdm_notebook
import glob
import matplotlib.pyplot as plt
import numpy as np
from lshash2 import LSHash

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

def cosine_similar(test, vectors):
    temp = {}
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for k,v in vectors.items():
        temp[k] = cos(test.unsqueeze(0), v.unsqueeze(0))
    output = {k: v for k, v in sorted(temp.items(), key=lambda item: item[1],
                                    reverse=True)}
    return output
          
            
#== test =================================================================
files = glob.glob(r'./data/handm_ko_kr/handm_ko_kr_raw/*.jpg')
path = './data/handm_ko_kr/handm_ko_kr_raw/'
outfit_path = './data/handm_ko_kr/handm_ko_kr_outfit/'
model_path = './models/'
vecs = {}

for f in tqdm(files, position=0, leave=True):
    vecs[re.search('(?<=\\\).*', f).group(0)] = get_feature_vector(f)

# save
np.save(model_path+"mytheresa_raw_resnet50_avgpool.npy", vecs, allow_pickle=True)

#== call =================================================================
path = './data/all_raw/'
outfit_path = './data/mytheresa/mytheresa_outfit/'

vecs = np.load(model_path+"mytheresa_raw_resnet50_avgpool.npy", allow_pickle=True)[()]
vecs2 = np.load(model_path+"handm_ko_kr_raw_resnet50_avgpool.npy", allow_pickle=True)[()]
vecs.update(vecs2)

# cosine similarity
sims = [{path+k: cosine_similar(v, vecs)} for k, v in list(vecs.items())[2500:2550]]

def similar_items(idx, sims, num):
    key = list(sims[idx].keys())[0]
    Image.open(key)
    
    plt.figure(figsize=(25,15))
    columns = 5
    for i, k in enumerate(list(sims[idx][key].keys())[0:num]):
        plt.subplot(columns / columns + 1, columns, i + 1)
        with open(path+k,'rb') as f:
            image=Image.open(f)
            plt.imshow(image)
    
    plt.figure(figsize=(25,15))
    columns = 5
    for i, k in enumerate(list(sims[idx][key].keys())[0:num]):
        plt.subplot(columns / columns + 1, columns, i + 1)
        try:
            with open(outfit_path+k,'rb') as f:
                image=Image.open(f)
                plt.imshow(image)
        except:
            with open(path+k,'rb') as f:
                image=Image.open(f)
                plt.imshow(image)
    # Check confidence level
    print(list(sims[idx][key].items())[0:10])

idx = 2 # 22, 26, 29
similar_items(idx, sims, 10)



# lsh similarity --- not as great!
k = 10 # hash size
L = 5  # number of tables
d = 512 # Dimension of Feature vector
lsh = LSHash(hash_size=k, input_dim=d, num_hashtables=L)
for img_path, vec in tqdm_notebook(vecs.items()):
    lsh.index(vec.flatten(), extra_data=img_path)

q_vec = list(vecs.values())[idx]
response = lsh.query(q_vec, num_results= 5)[3]
Image.open(path+response[0][1])






#== play =================================================================
testk = list(vecs.keys())[10]
testv = list(vecs.values())[10]
Image.open(path + testk)

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
test = {}
for k,v in vecs.items():
    test[k] = cos(testv.unsqueeze(0),
        v.unsqueeze(0))

testest = {k: v for k, v in sorted(test.items(), key=lambda item: item[1],
                         reverse=True)}

#for k, v in list(testest.items())[:10]:
#    img1 = Image.open(path+k)
#    img1.show()

Image.open(path + list(testest.items())[0][0])
Image.open(outfit_path + list(testest.items())[1][0])
Image.open(path + list(testest.items())[1][0])
Image.open(path + list(testest.items())[2][0])
Image.open(outfit_path + list(testest.items())[2][0])
Image.open(path + list(testest.items())[3][0])
Image.open(path + list(testest.items())[4][0])
Image.open(path + list(testest.items())[5][0])
Image.open(path + list(testest.items())[800][0])


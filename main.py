# -*- coding: utf-8 -*-

import numpy as np
from utils.similar_items import *
from utils.feature_vector import *


model_path = './models/'
path = './data/mytheresa/mytheresa_preprocessed/'
outfit_path = './data/mytheresa/mytheresa_outfit/'

    
#== call vectors =========================================================
vecs = np.load(model_path+"mytheresa_raw_resnet50_avgpool.npy", allow_pickle=True)[()]


#== play =================================================================
testk = list(vecs.keys())[1000] #'006.jpg'
testv = get_feature_vector(path+testk)

# cosine similarity
items_in_order = cosine_similarity(testv, vecs)

# display the most similar items
show_similar_items(testk, items_in_order, 5, path=path, outfit_path=outfit_path)

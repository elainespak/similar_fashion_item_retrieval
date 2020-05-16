# -*- coding: utf-8 -*-


import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


def cosine_similarity(test_vector, vectors):
    '''
    Measure cosine similarity
    '''
    temp = {}
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    for k,v in vectors.items():
        temp[k] = cos(test_vector.unsqueeze(0), v.unsqueeze(0))
    output = {k: v for k, v in sorted(temp.items(), key=lambda item: item[1],
                                    reverse=True)}
    return output


def show_similar_items(idx, sims, num, path, outfit_path):
    '''
    Grab and display top n(num) similar items
    '''
    images = list(sims.keys())[:10]
    plt.figure(figsize=(25,25))
    columns = 5
    for i, image in enumerate(images):
        plt.subplot(len(images) / columns + 1, columns, i + 1)
        # Check confidence level
        plt.xlabel('Confidence level: '+ str(sims[image][0]))
        img = Image.open(path+image)
        plt.imshow(img)

    plt.figure(figsize=(25,25))
    columns = 5
    for i, image in enumerate(images):
        try:
            plt.subplot(len(images) / columns + 1, columns, i + 1)
            img = Image.open(outfit_path+image)
            plt.imshow(img)
        except:
            pass


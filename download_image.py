import numpy as np 
import pandas as pd 
import sys, requests, shutil, os
from urllib import request, error
import sys

data=pd.read_csv('train.csv')
data.head(5)
def split(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
data=pd.read_csv('train.csv')
data.head(5)

def fetch_image(path):
    url=path
    response=requests.get(url, stream=True)
    with open('images/train_images/image.jpg', 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
links=data['url']
ids = data['landmark_id']
i=0

links=data['url']
linkslist = split(links,20)
ids = data['landmark_id']
idslist = split(ids,20)
i=0
for j in range(0,20):
    print(j)
    links = linkslist[j]
    ids = idslist[j]
    for link,id_ in zip(links,ids):              #looping over links to get images
        if os.path.exists('train/'+str(i)+'.jpg'):
            i+=1
            continue
        fetch_image(link)
        if(not os.path.isdir('images/train_images/'+str(id_))):
        	os.mkdir('images/train_images/'+str(id_))
        os.rename('images/train_images/image.jpg','images/train_images/'+str(id_)+'/'+ str(i)+'_'+str(id_)+ '.jpg')
        i+=1
        #if(i==15):   #uncomment to test in your machine
        #    break

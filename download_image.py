import numpy as np 
import pandas as pd 
import sys, requests, shutil, os
from urllib import request, error

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

for link,id_ in zip(links,ids):              #looping over links to get images
    if os.path.exists('train/'+str(i)+'.jpg'):
        i+=1
        continue
    fetch_image(link)
    os.rename('images/train_images/image.jpg','images/train_images/'+ str(i)+'_'+str(id_)+ '.jpg')
    i+=1
    #if(i==15):   #uncomment to test in your machine
    #    break

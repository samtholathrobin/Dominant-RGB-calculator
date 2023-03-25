from PIL import Image
import numpy as np
import scipy
from scipy import misc 
from scipy import cluster
import binascii as bsc
import time
# from fastapi import FastAPI
st=time.time()
# app=FastAPI()
img = Image.open("C:/Users/samth/Documents/Python Scripts/img_to_rgb/rocket.png")
l,b=img.size
img=img.resize((int(l/4), int(b/4)))
ar=np.asarray(img)
shape=ar.shape
ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

#finding clusters
codes, dist = scipy.cluster.vq.kmeans(ar, 5)

vecs, dist = scipy.cluster.vq.vq(ar, codes)       
counts, bins = np.histogram(vecs, len(codes)) 
idx_max=np.argmax(counts)
peak=codes[idx_max]
print('r: %f\tg: %f\tb: %f' %(peak[0],peak[1],peak[2]))
et=time.time()
print("time: ",et-st,"ms")
# @app.get("/")
# def read_root(): 
#     return {"message":(peak[0],peak[1],peak[2])}
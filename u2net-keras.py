from keras.models import load_model
import cv2
import numpy as np
import sys


input = sys.argv[1]
output = sys.argv[2]

model = './u2netp_keras.h5'


# load model
u2netp_keras = load_model(model, compile=False)

# load image
image = cv2.imread(input)

# convert to array
blob = cv2.dnn.blobFromImage(image, 1.0/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)

# predict
d1,d2,d3,d4,d5,d6,d7 = u2netp_keras.predict(blob)
pred = np.array(d1[:,0,:,:])[0]


# normalize
ma = np.max(pred)
mi = np.min(pred)
pred = (pred-mi)/(ma-mi)

pred = pred.squeeze()

pred = (pred*255).astype(np.uint8)

mask = cv2.resize(pred, image.shape[1::-1], interpolation=cv2.INTER_CUBIC)

# create alpha channel
b, g, r = cv2.split(image)
out = cv2.merge((b, g, r, mask))

# crop image
y, x = out[:,:,3].nonzero()
minx = np.min(x)
miny = np.min(y)
maxx = np.max(x)
maxy = np.max(y)
out = out[miny:maxy, minx:maxx]

# save image
cv2.imwrite(output, out)
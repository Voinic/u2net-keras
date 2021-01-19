from keras.models import load_model
import cv2
import numpy as np
import sys


input = sys.argv[1]
output = sys.argv[2]

model = './u2net_portrait_keras.h5'


# load model
u2netp_keras = load_model(model, compile=False)

# load image
image = cv2.imread(input)

# normalize input image
input = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

tmpImg = np.zeros((512, 512, 3))
input = input/np.max(input)

tmpImg[:,:,0] = (input[:,:,2]-0.406)/0.225
tmpImg[:,:,1] = (input[:,:,1]-0.456)/0.224
tmpImg[:,:,2] = (input[:,:,0]-0.485)/0.229

# convert BGR to RGB
tmpImg = tmpImg.transpose((2, 0, 1))
tmpImg = tmpImg[np.newaxis,:,:,:]

# predict
d1,d2,d3,d4,d5,d6,d7 = u2netp_keras.predict(tmpImg)
pred = np.array(1.0 - d1[:,0,:,:])[0]


# normalize
ma = np.max(pred)
mi = np.min(pred)
pred = (pred-mi)/(ma-mi)

pred = pred.squeeze()

pred = (pred*255).astype(np.uint8)

out = cv2.resize(pred, image.shape[1::-1], interpolation=cv2.INTER_CUBIC)

# save image
cv2.imwrite(output, out)
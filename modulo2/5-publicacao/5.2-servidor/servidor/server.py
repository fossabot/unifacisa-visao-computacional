from flask import Flask, request, Response
import jsonpickle
import numpy as np
import cv2
import argparse
import sys
import os
import json
import base64
import zlib


from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

opt_img_size=416
opt_conf_thres=0.3
opt_nms_thres=0.5
opt_half=True
opt_device=''
opt_cfg='conf/especializacao.cfg'
opt_weights='conf/best.pt'
classes = ['maca', 'laranja', 'banana']

# Start web server
app = Flask(__name__)

img_size = (320, 192) if ONNX_EXPORT else opt_img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
weights, half=opt_weights, opt_half
# Initialize
device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt_device)
# Initialize model
model = Darknet(opt_cfg, img_size)

# Load weights
if weights.endswith('.pt'):  # pytorch format
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
else:  # darknet format
    _ = load_darknet_weights(model, weights)

# Eval mode
model.to(device).eval()

# Half precision
half = half and device.type != 'cpu'  # half precision only supported on CUDA
if half:
    model.half()


@app.route('/api/predict', methods=['POST', 'GET'])
def predict():
    req = request
    nparr = np.fromstring(req.data, np.uint8)
    img0 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Padded resize
    img, *_ = letterbox(img0, new_shape=img_size)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
    img = np.ascontiguousarray(img, dtype=np.float16 if half else np.float32)  # uint8 to fp16/fp32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0

    # Get detections
    img = torch.from_numpy(img).to(device)
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred, _ = model(img)

    if opt_half:
        pred = pred.float()

    out_dict = {'predictions': []}
    for i, det in enumerate(non_max_suppression(pred, opt_conf_thres, opt_nms_thres)):  # detections per image
        if det is not None and len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *coord, conf, _, cls in det:
                boxini=(('%g ' * 6 + '\n') % (*coord, cls, conf))
                box=boxini.split()
                xmin, ymin, xmax, ymax, score =int(box[0]) ,int(box[1]) ,int(box[2]) ,int(box[3]) ,float(box[5]) 
                bbox = [xmin, ymin, xmax, ymax]
                out_dict['predictions'].append(
                        {
                            'scores': float(score),
                            'label': classes[int(cls.item())],
                            'boxes': bbox,
                            })
               
    out = json.dumps(out_dict)
    return str(out)    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081)
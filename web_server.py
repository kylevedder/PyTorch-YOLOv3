#!/usr/bin/env python3
import matplotlib
matplotlib.use('Agg')
from flask import Flask, request
import pickle
from models import *
from utils.utils import *
from utils.datasets import *
from timeit import default_timer as timer

import os
import sys
import time
import datetime
import argparse
import cv2

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import joblib

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

parser = argparse.ArgumentParser()
parser.add_argument("--hostname", type=str, default="localhost", help="Server hostname")
parser.add_argument("--model_def", type=str, default="config/yolov3-tiny.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="weights/yolov3-tiny.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("output", exist_ok=True)

# Set up model
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

model.eval()  # Set in evaluation mode

classes = load_classes(opt.class_path)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def resize_and_pad_image(img, desired_size):
        old_size = img.shape[:2]
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img = cv2.resize(img, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)

        color = [0, 0, 0]
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
            value=color)


def detect_image(img):
    # Configure input
    input_imgs = Variable(img.type(Tensor))

    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

    return detections

def draw_detections(img, detections_lst, output_path):
    assert(type(detections_lst) == list)
    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)

    class_lst = []

    # Draw bounding boxes and labels of detections
    for detections in detections_lst:
        if detections is None:
            continue
        # Rescale boxes to original image
        detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            if cls_conf.item() < opt.conf_thres:
                continue

            x1 = max(x1, 0)
            y1 = max(y1, 0)
            x2 = min(x2, img.shape[0])
            y2 = min(y2, img.shape[1])

            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 20 or box_h < 20:
                continue

            class_lst.append(classes[int(cls_pred)])
            print("\t+ Label: %s, (%d, %d) Conf: %.5f" % (classes[int(cls_pred)], box_w, box_h, cls_conf.item()))

            color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            # Create a Rectangle patch
            bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
            # Add the bbox to the plot
            ax.add_patch(bbox)
            # Add label
            plt.text(
                x1,
                y1,
                s=classes[int(cls_pred)],
                color="white",
                verticalalignment="top",
                bbox={"color": color, "pad": 0},
            )

    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
    plt.close()
    plt.clf()
    return class_lst

###########################################
# Flask setup
###########################################

app = Flask(__name__)

call_idx = 0

@app.route('/', methods=['POST'])
def result():
    global call_idx
    start = timer()
    webcam = pickle.loads(request.data)
    webcam = np.flip(webcam, 2).copy()
    renderable_webcam = webcam.copy()
    webcam = webcam / 255
    webcam = resize_and_pad_image(webcam, opt.img_size)
    webcam = webcam.swapaxes(0, 1)
    webcam = webcam.swapaxes(0, 2)
    webcam = np.expand_dims(webcam, axis=0)        
    webcam = torch.from_numpy(webcam)
    detections = detect_image(webcam)
    class_lst = draw_detections(renderable_webcam, detections, f"webcam_output/webcam{call_idx}.png")
    call_idx += 1
    end = timer()
    
    print("Delta t: ", end - start)
    return "Success: " + ", ".join(class_lst)

app.run(opt.hostname, debug=False, port=5000)
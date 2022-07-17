# Add code to sys.path
import sys
sys.path.append('docExtractor/src')


# Select GPU ID
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
from models import load_model_from_path
from utils import coerce_to_path_and_check_exist
# from utils.path import MODELS_PATH
# from utils.constant import MODEL_FILE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TAG = 'default'
model_path = coerce_to_path_and_check_exist('models/doc-extractor/default/model.pkl')
model, (img_size, restricted_labels, normalize) = load_model_from_path(model_path, device=device, attributes_to_return=['train_resolution', 'restricted_labels', 'normalize'])
_ = model.eval()


from torch_snippets import *
from PIL import Image
import numpy as np
from utils.image import resize
# from utils.constant import LABEL_TO_COLOR_MAPPING
# from utils.image import LabeledArray2Image
    
from torch_snippets import resize as tsresize

def get_bbs(pred, img_size):
    imgray = (pred == 2).astype(np.uint8)*255
    imgray = tsresize(imgray, img_size[:2])
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    bbs = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        bbs.append((x, y-10, x+w, y+h+10))
    return bbs

def get_bboxes(fpath):
    fpath = str(fpath)
    line(fpath)
    img = read(fpath, 1)
    og_img_size = img.shape
    img = Image.fromarray(img)

    img = resize(img, img_size)
    inp = np.array(img, dtype=np.float32) / 255
    if normalize:
        inp = ((inp - inp.mean(axis=(0, 1))) / (inp.std(axis=(0, 1)) + 10**-7))
        
    inp = torch.from_numpy(inp.transpose(2, 0, 1)).float().to(device)
    print(inp.shape)
    # compute prediction
    pred = model(inp.reshape(1, *inp.shape))
    pred = pred[0]
    pred = pred.max(0)[1].cpu().numpy()
    # show(img)
    # show(pred)
    bbs = get_bbs(pred, og_img_size)
    bbs = bbfy(bbs)
    bbs = [bb for bb in bbs if bb.w > 20]
    # show(read(fpath, 1), bbs=bbs, texts=[bb.w for bb in bbs])
    save_path=f'bbs/{stem(fpath)}.bbs'
    dumpdill(bbs, save_path)
    return save_path

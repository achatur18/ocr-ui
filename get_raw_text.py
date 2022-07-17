import sys
sys.path.append('docExtractor/src')

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests 
from PIL import Image
from torch_snippets import *

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-handwritten")


# load image from the IAM dataset 
def trocr(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values, output_scores=True, return_dict_in_generate=True)
    confs = []
    for x in generated_ids.scores:
        x = torch.exp(x)
        x = x/torch.sum(x)
        confs.append(x.max().item())
    conf = min(confs[:-1])
    generated_text = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0] 
    return generated_text, conf


import time
from create_bb import get_bboxes

def raw_text(fpath, bb_path=None):
        try:
            line(str(fpath))    
            start = time.time()

            if bb_path==None:
                bb_path=get_bboxes(fpath)

            image = read(fpath, 1)
            bbs = bbfy(loaddill(bb_path))
            # bbs = [bb for bb in bbs if bb.w > ]
            texts = []
            confs = []
            bbs = np.array(bbs).clip(0, 1000000)
            for bb in Tqdm(bbs):
                crop = crop_from_bb(image, bb)
                crop = Image.fromarray(crop)
                # is this crop var is typed or handwritten
                text, conf = trocr(crop)
                texts.append(text)
                confs.append(conf)
            texts=" ".join(texts)
            return texts
        except Exception as e:
            print(e)
            return ""

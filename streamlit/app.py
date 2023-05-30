import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time

import torch

from model import EAST
from detect import detect

st.set_page_config(initial_sidebar_state="collapsed")
st.title("OCR with CV-09 Team model")
def load_model(model_file):
    model = EAST(pretrained=False).to('cpu')
    model.load_state_dict(torch.load(model_file, map_location='cpu'))
    model.eval()

    return model

def do_inference(model, img, input_size=2048):

    image_fnames, by_sample_bboxes = [], []

    
    # img = torch.tensor(img, dtype=torch.float32).permute(2,0,1).unsqueeze(dim=0)
    
    image_fnames, by_sample_bboxes = ['test'], []
    images = []
    images.append(img)
    
    by_sample_bboxes.extend(detect(model, images, input_size))
    #st.text(by_sample_bboxes)
    ufo_result = dict(images=dict())
    for image_fname, bboxes in zip(image_fnames, by_sample_bboxes):
        words_info = {idx: dict(points=bbox.tolist()) for idx, bbox in enumerate(bboxes)}
        ufo_result['images'][image_fname] = dict(words=words_info)

    return ufo_result

def main():
    
    uploaded_model = st.file_uploader("Upload your model.", accept_multiple_files=False, type=['pth', 'pt'])
    if uploaded_model is not None:
        model = load_model(uploaded_model)
        model.eval()
        
    
    uploaded_image = st.file_uploader("Choose an image...", accept_multiple_files=False)
    if uploaded_image is not None and uploaded_model is not None:
        img = np.array(Image.open(uploaded_image))
        fin_img = img.copy()
        
        ufo_result = do_inference(model, img)
        st.text(img.shape)
        
        for _, v in ufo_result['images']['test']['words'].items():
            v = v['points']
            v.append(v[0])
            cv2.polylines(fin_img, [np.array(v, dtype=np.int32)], True, (0, 0, 255), 1)
        st.image(fin_img)
        
if __name__ == '__main__':
    main()
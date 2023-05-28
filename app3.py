#The usual suspects
import numpy as np
import pandas as pd
import pickle
import datetime
import queue
from typing import List, NamedTuple
import threading

#Model handling
import torch
from torchvision.models import resnet50
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Module, Dropout, Identity
from torchvision import transforms

#Streamlit-related
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

#Image and video data handling
import av
import cv2
import imutils
from PIL import Image

st.set_page_config(page_title="FMCG Food Items Recognition Model", page_icon="🧇")

CONFIGS = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "IMG_MEAN": [0.485, 0.456, 0.406],
    "IMG_STD": [0.229, 0.224, 0.225],
}

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def load_label_encoder():
    le_total = pickle.loads(open(r"C:\Users\ELIZABETH CHENG\source\repos\image_app\img_classf\model\le_total.pickle", "rb").read())
    return le_total

# model class
class ObjectDetector(Module):
    def __init__(self, baseModel, numClasses_total):
        super(ObjectDetector, self).__init__()
        # initialize the base model and the number of classes
        self.baseModel = baseModel
        self.numClasses_total = numClasses_total
        # build the regressor head for outputting the bounding box coordinates
        self.regressor = Sequential(          
            Linear(baseModel.fc.in_features, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 4),
            Sigmoid()
        )
        # build the classifier head to predict the class labels for halal
        self.classifier_total = Sequential(
            Linear(baseModel.fc.in_features, 512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            ReLU(),
            Dropout(),
            Linear(512, self.numClasses_total)
        )
        # set the classifier of our base model to produce outputs
        # from the last convolution block
        self.baseModel.fc = Identity()

    def forward(self, x):
        # pass the inputs through the base model and then obtain
        # predictions from different branches of the network
        features = self.baseModel(x)
        bboxes = self.regressor(features)
        classLogits_total = self.classifier_total(features)
        # return the outputs as a tuple
        return (bboxes, classLogits_total)

# load our object detector, set it evaluation mode
def load_model():
    # model = ObjectDetector()
    le_total = load_label_encoder()
    resnet = resnet50(pretrained=True)
    model = ObjectDetector(resnet, len(le_total.classes_))
    
    model.load_state_dict(torch.load(r"C:\Users\ELIZABETH CHENG\source\repos\image_app\img_classf\model\model_state.pt",map_location=torch.device('cpu')))
    model.eval()
    return model

# Load label encoder and model
le_total = load_label_encoder()
model = load_model()
transforms_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean=CONFIGS['IMG_MEAN'], std=CONFIGS['IMG_STD'])
])

format_list = ["Video Stream", "File Upload"]
with st.sidebar:
    format_name = st.selectbox("Select your recognition mode:", format_list)

if format_name == format_list[0]:
    if "page" not in st.session_state:
        st.session_state.page = 0

    def nextpage(): st.session_state.page += 1
    def restart(): st.session_state.page = 0

    placeholder = st.empty()
    st.button("Next",on_click=nextpage,disabled=(st.session_state.page > 3))

    if st.session_state.page == 0:

        st.title("Welcome to the live video feed recognition mode! 🧇")
        st.text("Insert instructions here")

    elif st.session_state.page == 1: 
        #conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
        ice_complete_time = datetime.datetime.now()
        label_value = 0

        class Detection(NamedTuple):
            label: str
            conf: float
            b_box: tuple
        result_q: "queue.Queue[List[Detection]]" = queue.Queue()

        def video_frame_callback(frame: av.VideoFrame): 
            global ice_complete_time
            if webrtc_ctx.state.playing and ice_complete_time is None:
                ice_complete_time = datetime.datetime.now()
                print("ICE connection state complete time is:", ice_complete_time)
            now = datetime.datetime.now()
            while ice_complete_time+datetime.timedelta(0, 4) > now:
                frame = frame.to_image()
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                frame = imutils.resize(frame, width=400)
                orig = frame.copy()
                frame = cv2.resize(frame, (224, 224))
                frame = frame.transpose((2, 0, 1))  
                frame = torch.from_numpy(frame)
                frame = transforms_test(frame).to(CONFIGS['DEVICE'])
                frame = frame.unsqueeze(0)
                # run inference
                (boxPreds, labelPreds_total) = model(frame)
                (startX, startY, endX, endY) = boxPreds[0]
                # determine the class label with the largest predicted probability
                labelPreds_total = torch.nn.Softmax(dim=-1)(labelPreds_total)
                i_total = labelPreds_total.argmax(dim=-1).cpu()
                label_total = le_total.inverse_transform(i_total)[0]
                label = label_total

                with open(r"C:\Users\ELIZABETH CHENG\source\repos\image_app\img_classf\writer\Label.txt", "w") as text_file:
                    text_file.write(f"{label}")
                
                orig = imutils.resize(orig, width=600)
                (h, w) = orig.shape[:2]
                startX = int(startX * w)
                startY = int(startY * h)
                endX = int(endX * w)
                endY = int(endY * h)

                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(orig, (startX, startY), (endX, endY),
                (0, 255, 0), 2)

                #put frames into writer folder
                filepath = r"C:\Users\ELIZABETH CHENG\source\repos\image_app\img_classf\writer\Img.png"
                cv2.imwrite(filepath, orig)

                cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 2)

                orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
                orig = Image.fromarray(orig)

                #put detections from this round into queue
                detection = Detection(
                    label = label, 
                    conf = 0,
                    b_box = (startX, startY, endX, endY)
                    )
                result_q.put(detection)
                return av.VideoFrame.from_image(orig)
            else: 
                #TODO: 
                #Add the drawn on with the most frequent label
                pass
        
        st.write("If the video capture is satisfactory, move on to view the results!", value = False)
        labels_placeholder = st.empty()
        key_lst = []
        webrtc_ctx = webrtc_streamer(
            key="object-detection",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=RTC_CONFIGURATION,
            async_processing=True,
        )
        result = []
        if webrtc_ctx.state.playing:
            #labels_placeholder = st.empty()
            while True:
                if len(key_lst) == 0: 
                    key_lst.append(0)
                else:
                    key_lst.append(key_lst[-1]+1)
                result.append(result_q.get()[0])
                ##display the most frequent value at the time of reading the frame
                label_value = max(set(result), key=result.count)
                #label_value = get_max_label(result)
                #st.write(key_lst[-1])
                labels_placeholder.text_input("Object Label:", f"{label_value}", key = key_lst[-1])
    elif st.session_state.page == 2: 
        with open(r"C:\Users\ELIZABETH CHENG\source\repos\image_app\img_classf\writer\Label.txt") as f:
            contents = f.read()
        st.text_input("Label of the object:", f"{contents}")
        st.image(r"C:\Users\ELIZABETH CHENG\source\repos\image_app\img_classf\writer\Img.png")
        st.button("Restart the video capture?", on_click = restart)
    
if format_name == format_list[1]: 
    st.file_uploader("Upload Your Photos Here:")

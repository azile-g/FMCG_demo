#The usual suspects
import numpy as np
import pickle
from datetime import datetime 
import queue
from typing import List, NamedTuple
import os
import logging
import glob
import os, shutil

#Model handling
import torch
from torchvision.models import resnet50
from torch.nn import Sequential, Linear, ReLU, Sigmoid, Module, Dropout, Identity
from torchvision import transforms
from torch.nn import BatchNorm1d

#Streamlit-related
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import streamlit_toggle as toggle
from twilio.rest import Client

#Image and video data handling
import av
import cv2
import imutils
from PIL import Image

from my_utils import helper
#from myutils import gsotr_util

st.set_page_config(page_title="FMCG Food Items Recognition Model", page_icon="🧇")

logger = logging.getLogger(__name__)

CONFIGS = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "IMG_MEAN": [0.485, 0.456, 0.406],
    "IMG_STD": [0.229, 0.224, 0.225],
}

def get_ice_servers():
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google." 
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]
    client = Client(account_sid, auth_token)
    token = client.tokens.create()
    return token.ice_servers

@st.cache_resource
def load_label_encoder():
    le_total = pickle.loads(open(r"le_total.pickle", "rb").read())
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
            BatchNorm1d(128),
            ReLU(),
            Linear(128, 64),
            BatchNorm1d(64),
            ReLU(),
            Linear(64, 32),
            BatchNorm1d(32),
            ReLU(),
            Linear(32, 4),
            Sigmoid()
        )
        # build the classifier head to predict the class labels for halal
        self.classifier_total = Sequential(
            Linear(baseModel.fc.in_features, 512),
            BatchNorm1d(512),
            ReLU(),
            Dropout(),
            Linear(512, 512),
            BatchNorm1d(512),
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
@st.cache_resource
def load_model():
    # model = ObjectDetector()
    le_total = load_label_encoder()
    resnet = resnet50(pretrained=True)
    model = ObjectDetector(resnet, len(le_total.classes_))
    
    model.load_state_dict(torch.load(r"model_state.pt",map_location=torch.device('cpu')))
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

def write_timestamp(label_to_log): 
    text_to_log = f"{label_to_log}: "+f"{datetime.now()}"
    with open(r"writer\\"+f"tmp\\{label_to_log}_timestamp.txt", "w") as text_file:
        text_file.writelines(f"{text_to_log}\n")
    return

if "page" not in st.session_state:
    st.session_state.page = 0
def nextpage(): 
    st.session_state.page += 1
def restart(): 
    st.session_state.page = 0

if st.session_state.page == 0:
    class Detection(NamedTuple):
        label: str
        conf: float
        b_box: tuple
    result_q: "queue.Queue[List[Detection]]" = queue.Queue()

    def video_frame_callback(frame: av.VideoFrame): 
        frame = frame.to_image()
        frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
        frame = imutils.resize(frame, width=400)
        orig = frame.copy()

        #put frames into writer folder
        now = datetime.now()
        file_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_{now.microsecond}"
        filepath = r"writer\\"+f"tmp\{file_name}.png"
        cv2.imwrite(filepath, orig)

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
        
        orig = imutils.resize(orig, width=600)
        (h, w) = orig.shape[:2]
        startX = int(startX * w)
        startY = int(startY * h)
        endX = int(endX * w)
        endY = int(endY * h)

        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(orig, (startX, startY), (endX, endY),
        (0, 255, 0), 2)

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
    
    st.write("Take out your items and let us know when you are ready!")
    labels_placeholder = st.empty()
    key_lst = []
    video_toggle = toggle.st_toggle_switch(label="Video Recognition", 
                key="Key1", 
                default_value=False, 
                label_after = False, 
                inactive_color = '#D3D3D3', 
                active_color="#11567f", 
                track_color="#29B5E8"
                )

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        desired_playing_state=video_toggle,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        #rtc_configuration={"iceServers": get_ice_servers()},
        async_processing=True,
    )
    now = datetime.now()
    st.session_state.txt_file_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_{now.microsecond}"
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
            labels_placeholder.text_input("Most Frequent Object Label (at point of capture):", f"{label_value}", key = key_lst[-1])
            with open(r"writer\\"+f"tmp\{st.session_state.txt_file_name}_Most_Freq_Label.txt", "w") as text_file:
                text_file.write(f"{label_value}")
            with open(r"writer\\"+f"tmp\{st.session_state.txt_file_name}_All_Label.txt", "w") as text_file:
                for i in result:
                    text_file.writelines(f"{i}\n")
    if video_toggle == True: 
        write_timestamp("start_video")
    if video_toggle == False: 
        #write_timestamp("stop_video")
        #labels_placeholder = st.empty()
        try: 
            with open(r"writer\\"+f"tmp\{st.session_state.txt_file_name}_Most_Freq_Label.txt", "r") as text_file:
                contents = text_file.read()
            labels_placeholder.text_input("Most Recent Object Label Logged:", f"{contents}")
        except: 
            labels_placeholder.text_input("Most Recent Object Label Logged:", "")

elif st.session_state.page == 1: 

    with st.form("relabel", clear_on_submit=False): 
        try: 
            _, label_path = helper.reader_paths("/writer/tmp/", "*.txt", name="Freq")
            with open(label_path) as f:
                contents = f.read()
            _, latest_img = helper.reader_paths(f"/writer/tmp/", "*.png")
            print(latest_img)
            img = cv2.imread(latest_img)
            img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            img = cv2.putText(img, contents, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            st.write("Current capture:")
            st.image(img)
            st.text_input("Most Frequent Label of the object:", f"{contents}")
            prdt, weight, halal, healthy = contents.split("_")
            print(prdt, weight, halal, healthy)
            prdt_values = [
                'Babyfood', 'Babymilk-powder-', 'BeehoonVermicelliMeesua',
                'BiscuitsCrackersCookies', 'Cookies', 'Crackers', 'FlavoredMilk',
                'Flour', 'HoneyOtherSpreads', 'InstantMeals', 'InstantNoodles',
                'Kaya', 'MaternalMilkPowder', 'Milo-powder-', 'Nonfood',
                'NutellaChocolate', 'Nuts', 'Oil', 'OtherBakingNeeds',
                'OtherNoodles', 'OtherSauceDressingSoupbasePaste', 'Pasta',
                'Peanutbutter', 'PotatochipsKeropok', 'RolledOatsInstantOatmeal',
                'Salt', 'Sardines', 'Sugar', 'SweetsChocolatesOthers',
                'Tea-powder-leaves-'
                ]
            prdt_idx = prdt_values.index(prdt)
            weight_values = [
                '1-99g', '100-199g', '1000-1999g', '200-299g', '300-399g',
                '3000-3999g', '400-499g', '500-599g', '600-699g', '700-799g',
                '800-899g', '900-999g', 'Nonfood'
                ]
            weight_idx = weight_values.index(weight)
            halal_values = ['Halal', 'NonHalal', 'Nonfood']
            halal_idx = halal_values.index(halal)
            healthy_values = ['Healthy', 'NonHealthy', 'Nonfood']
            healthy_idx = healthy_values.index(healthy)
            product_dd = st.selectbox("Relabel the product type:",
                                    prdt_values, 
                                    index=prdt_idx
                                    )
            weight_dd = st.selectbox("Relabel the product weight:", 
                                    weight_values, 
                                    index=weight_idx
                                    )
            Halal = st.selectbox("Relabel the product's Halal certification:", 
                                halal_values, 
                                index=halal_idx)
            ##healthy not healthy
            healthy_dd = st.selectbox("Relabel the product's Halal certification:", 
                                healthy_values, 
                                index=healthy_idx)
        except: 
            test, test1 = helper.reader_paths("/writer/tmp/", "*.txt", name="Freq")
            st.write(test1)
            _, latest_img = helper.reader_paths(f"/writer/tmp/", "*.png")
            print(latest_img)
            pass

        submitted = st.form_submit_button("Submit")
        if submitted:
            submit_ts = datetime.now()
            user_label = product_dd + "_" + weight_dd + "_" + Halal + "_" + healthy_dd
            with open(r"writer\tmp\User_Label.txt", "w") as text_file:
                text_file.write(f"{user_label}")
            st.info(f"User Label Logged: {user_label}")
            write_timestamp("submit_button")
            now = datetime.now()
            folder_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}_{now.microsecond}"
            os.rename("./writer/tmp", f"./writer/{folder_name}")
            os.mkdir(f"./writer/tmp")
            # TODO datetime stamp, user logged label
            # TODO save locally 
            #folder = 'writer'
            #for filename in os.listdir(folder):
            #    file_path = os.path.join(folder, filename)
            #    try:
            #        if os.path.isfile(file_path) or os.path.islink(file_path):
            #            os.unlink(file_path)
            #        elif os.path.isdir(file_path):
            #            shutil.rmtree(file_path)
            #    except Exception as e:
            #        print('Failed to delete %s. Reason: %s' % (file_path, e))

next_button = st.button("Next",
                        on_click=nextpage,
                        disabled=(st.session_state.page >= 1))
if next_button: 
    write_timestamp("next_button")
restart_button = st.button("Restart", 
                            on_click=restart, 
                            disabled=(st.session_state.page < 1))
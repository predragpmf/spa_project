import argparse
import pickle
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import datetime
import time

import face_recognition
from PIL import Image

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# Create directories if they don't already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)

parser = argparse.ArgumentParser(description="Recognize faces in an image")
parser.add_argument("--train", action="store_true", help="Train on input data")
parser.add_argument("--test", action="store_true", help="Test the model with an unknown image")
parser.add_argument("-m", action="store", default="hog", choices=["hog", "cnn"], 
                    help="Which model to use for training: hog (CPU), cnn (GPU)")
parser.add_argument("-f", action="store", help="Path to an image with an unknown face")
args = parser.parse_args()


def encode_known_faces(model: str = "hog",
                       encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    
    names = []
    encodings = []

    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        face_locations = face_recognition.face_locations(image, model=model)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)


def recognize_faces(video_location: str, model: str = "hog",
    encodings_location: Path = DEFAULT_ENCODINGS_PATH) -> None:
    
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
        
    cap = cv2.VideoCapture(video_location)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = int(time.time())
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_counter = 0
    name_set = set()
    with open('output/output.log', 'a') as f:  
        f.write("date;real_time;video_time;name\n")
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_counter += 1
                scale_percent = 50
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                input_image = np.asarray(rgb_frame)
        
                input_face_locations = face_recognition.face_locations(input_image, 
                                                                       model=model)
                input_face_encodings = face_recognition.face_encodings(input_image, 
                                                                       input_face_locations)
                pillow_image = Image.fromarray(input_image)
                #draw = ImageDraw.Draw(pillow_image)
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                elapsed_time = int(current_frame / fps)
                current_time = start_time + elapsed_time
                current_datetime = datetime.datetime.fromtimestamp(current_time).strftime("%Y-%m-%d;%H:%M:%S")
                if (frame_counter == 10):
                    percentage = int((current_frame / total_frames) * 100)
                    print(str(percentage) + "%")
                    frame_counter = 0
                    
                for unknown_encoding in input_face_encodings:
                    name = _recognize_face(unknown_encoding, loaded_encodings)
                    if not name:
                        print("Unknown face")
                        name = "Unknown"
                        name_set.clear()
                        continue
                    if name in name_set:
                        continue
                    name_set.add(name)
                    print(name)
                    td = datetime.timedelta(seconds=elapsed_time)
                    f.write(current_datetime + ";" + str(td) + ";" + name + '\n')
                    #_display_face(draw, bounding_box, name)
                    #pillow_image.show()
                
                #del draw
            else:
                print("Processing finished.")
                break

def _recognize_face(unknown_encoding, loaded_encodings):
    
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"],
                                                     unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match)
    if votes:
        return votes.most_common(1)[0][0]


def _display_face(draw, bounding_box, name):
    
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(((text_left, text_top), (text_right, text_bottom)),
                   fill=BOUNDING_BOX_COLOR, outline=BOUNDING_BOX_COLOR)
    draw.text((text_left, text_top), name, fill=TEXT_COLOR)
    

if __name__ == "__main__":
    if args.train:
        encode_known_faces(model=args.m)
    if args.test:
        recognize_faces(video_location=args.f, model=args.m)
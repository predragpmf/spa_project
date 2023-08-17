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
    """
    Loads images in the training directory and builds a dictionary of their
    names and encodings.
    """
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
    """
    Given an unknown video, get the locations and encodings of any faces and
    compares them against the known encodings to find potential matches.
    """
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
        
    cap = cv2.VideoCapture(video_location)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = int(time.time())
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_counter = 0
    name_set = set()
    with open('output/output.log', 'a') as f:  
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_counter += 1
                npArray = np.asarray(frame)
                img = Image.fromarray(npArray)
                img = img.convert('RGB')
                input_image = np.array(img)
                input_face_locations = face_recognition.face_locations(input_image, 
                                                                       model=model)
                input_face_encodings = face_recognition.face_encodings(input_image, 
                                                                       input_face_locations)
                pillow_image = Image.fromarray(input_image)
                #draw = ImageDraw.Draw(pillow_image)
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                elapsed_time = int(current_frame / fps)
                current_time = start_time + elapsed_time
                current_datetime = datetime.datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S")
                if (frame_counter == 10):
                    percentage = int((current_frame / total_frames) * 100)
                    print(str(percentage) + "%")
                    frame_counter = 0
                    
                for unknown_encoding in input_face_encodings:
                    name = _recognize_face(unknown_encoding, loaded_encodings)
                    if not name:
                        name = "Unknown"
                        name_set.clear()
                        continue
                    if name in name_set:
                        continue
                    name_set.add(name)
                    f.write(current_datetime + " - " + name + '\n')
                    #_display_face(draw, bounding_box, name)
                    #pillow_image.show()
                
                #del draw


def _recognize_face(unknown_encoding, loaded_encodings):
    """
    Given an unknown encoding and all known encodings, find the known
    encoding with the most matches.
    """
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"],
                                                     unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match)
    if votes:
        return votes.most_common(1)[0][0]


def _display_face(draw, bounding_box, name):
    """
    Draws bounding boxes around faces, a caption area, and text captions.
    """
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
import datetime
import time
import os
import shutil
from collections import Counter
from pathlib import Path
import pickle
import cv2
import numpy
import face_recognition

# Default paths:
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
DEFAULT_INPUT_PATH = Path("input/")
DEFAULT_ARCHIVE_PATH = Path("archive/")

# Create directories if they don't already exist
Path("input").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("archive").mkdir(exist_ok=True)


def recognize_faces(video_location: str, model: str):

    # Load face encodings from file
    with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # Open video for reading
    cap = cv2.VideoCapture(video_location)

    # Get statistics
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_time = int(time.time())
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_counter = 0
    name_set = set()

    # Open log for writing
    with open('output/output.log', 'a') as f:  
        f.write("date;real_time;video_time;name\n")

        while cap.isOpened():

            # Read video frame by frame:
            ret, frame = cap.read()

            # If frame exists
            if ret:
                frame_counter += 1

                # Make the frame smaller and RBG
                scale_percent = 50
                width = int(frame.shape[1] * scale_percent / 100)
                height = int(frame.shape[0] * scale_percent / 100)
                dim = (width, height)
                resized_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                input_image = numpy.asarray(rgb_frame)

                # Detect faces in frame
                input_face_locations = face_recognition.face_locations(input_image, 
                                                                       model=model)
                input_face_encodings = face_recognition.face_encodings(input_image, 
                                                                       input_face_locations)
                #pillow_image = Image.fromarray(input_image)
                current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
                elapsed_time = int(current_frame / fps)
                current_time = start_time + elapsed_time
                current_datetime = datetime.datetime.fromtimestamp(current_time).strftime("%Y-%m-%d;%H:%M:%S")
                if frame_counter == 10:
                    percentage = int((current_frame / total_frames) * 100)
                    print(str(percentage) + "%")
                    frame_counter = 0

                for unknown_encoding in input_face_encodings:
                    name = recognize_face(unknown_encoding, loaded_encodings)
                    if not name:
                        #print("Unknown face")
                        name = "Unknown"
                        name_set.clear()
                        continue
                    if name in name_set:
                        continue
                    name_set.add(name)
                    print(name)
                    td = datetime.timedelta(seconds=elapsed_time)
                    f.write(current_datetime + ";" + str(td) + ";" + name + '\n')
            else:
                print("Processing finished.")
                break


def recognize_face(unknown_encoding, loaded_encodings):

    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"],
                                                     unknown_encoding)
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match)
    if votes:
        return votes.most_common(1)[0][0]


if __name__ == "__main__":
    while True:
        # Read files from input
        video_files = os.listdir(DEFAULT_INPUT_PATH)
        video_formats = {"mp4", "avi", "mov", "mkv"}
        for video in video_files:
            format_extension = video.split(".")[-1]
            if format_extension in video_formats:
                # If video is available, process it, and move it to the archive
                video_path = os.path.join(DEFAULT_INPUT_PATH, video)
                recognize_faces(video_location=video_path, model="cnn")
                time.sleep(5)
                shutil.move(video_path, DEFAULT_ARCHIVE_PATH)
        time.sleep(1)

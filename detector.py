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
import argparse
import stats


# Default paths:
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
DEFAULT_INPUT_PATH = Path("input/")
DEFAULT_ARCHIVE_PATH = Path("archive/")

parser = argparse.ArgumentParser(description="Detect on input data")
parser.add_argument("-m", "--model", help="Choose model for detection: hog(CPU), cnn(GPU)")
args = parser.parse_args()

# Create directories if they don't already exist
Path("input").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("archive").mkdir(exist_ok=True)


def recognize_faces(video_location: str, model: str):

    # Load face encodings from file
    with DEFAULT_ENCODINGS_PATH.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)

    # Open video for reading
    video_capture = cv2.VideoCapture(video_location)

    # Get statistics
    frames_per_sec = video_capture.get(cv2.CAP_PROP_FPS)
    start_time_sec = int(time.time())
    total_frames = video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_counter = 0
    name_set = set()

    # Open log for writing
    with open('output/output.csv', 'a') as output_file:  
        output_file.write("date,real_time,video_time,name\n")

        while video_capture.isOpened():

            # Read video frame by frame:
            frame_exists, video_frame = video_capture.read()

            if frame_exists:
                frame_counter += 1

                # Make the frame smaller and RBG
                scale_percent = 50
                frame_width = int(video_frame.shape[1] * scale_percent / 100)
                frame_height = int(video_frame.shape[0] * scale_percent / 100)
                frame_dimensions = (frame_width, frame_height)
                resized_frame = cv2.resize(video_frame, frame_dimensions, interpolation = cv2.INTER_AREA)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                processed_frame = numpy.asarray(rgb_frame)

                # Detect faces in frame:
                input_face_locations = face_recognition.face_locations(processed_frame, 
                                                                       model=model)
                input_face_encodings = face_recognition.face_encodings(processed_frame, 
                                                                       input_face_locations)

                # Calculate log stats:
                current_frame_number = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
                elapsed_time_sec = int(current_frame_number / frames_per_sec)
                current_time_sec = start_time_sec + elapsed_time_sec
                current_datetime = datetime.datetime.fromtimestamp(current_time_sec).strftime("%Y-%m-%d,%H:%M:%S")

                # Every 10 frames print percentage completed:
                if frame_counter == 10:
                    percentage_complete = int((current_frame_number / total_frames) * 100)
                    print(str(percentage_complete) + "%")
                    frame_counter = 0
                if not input_face_encodings:
                    name_set.clear()
                # Process frames where face is detected:
                for unknown_encoding in input_face_encodings:
                    detected_name = recognize_face(unknown_encoding, loaded_encodings)
                    # If face is not recognized:
                    if not detected_name:
                        detected_name = "Unknown"
                        name_set.clear()
                        continue
                    if detected_name in name_set:
                        continue
                    name_set.add(detected_name)
                    print(detected_name)
                    video_time = datetime.timedelta(seconds=elapsed_time_sec)
                    output_file.write(current_datetime + "," + str(video_time) + "," + detected_name + '\n')
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

def video_stats():
    pass


if __name__ == "__main__":
    if args.model:
        while True:
            # Read files from input:
            video_files = os.listdir(DEFAULT_INPUT_PATH)
            video_formats = {"mp4", "avi", "mov", "mkv"}
            for video in video_files:
                format_extension = video.split(".")[-1]
                if format_extension in video_formats:
                    # If video is available, process it, and move it to the archive:
                    video_path = os.path.join(DEFAULT_INPUT_PATH, video)
                    recognize_faces(video_location=video_path, model=args.model)
                    stats.run()
                    os.remove("output/output.csv")
                    shutil.move(video_path, DEFAULT_ARCHIVE_PATH)
            time.sleep(1)
    else:
        print("Missing detection model: detector.py -m=model")

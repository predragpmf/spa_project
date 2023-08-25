from pathlib import Path
import face_recognition
import pickle
import argparse

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

parser = argparse.ArgumentParser(description="Train on input data")
parser.add_argument("-m", "--model", help="Choose model for training: hog(CPU), cnn(GPU)")
args = parser.parse_args()

Path("training").mkdir(exist_ok=True)

def encode_known_faces(model: str):
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
    with DEFAULT_ENCODINGS_PATH.open(mode="wb") as f:
        pickle.dump(name_encodings, f)
        

if __name__ == "__main__":
    if args.model:
        encode_known_faces(model=args.model)
    else:
        print("Missing training model")
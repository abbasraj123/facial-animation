import cv2
import numpy as np
import dlib
import os
import pathlib as plb
from project_paths import paths, parse_data_root, ensure_dirs

detector = dlib.get_frontal_face_detector() 
predictor_path = plb.Path(__file__).resolve().parent / "shape_predictor_68_face_landmarks.dat"
predictor= dlib.shape_predictor(str(predictor_path)) 
debug_frame_path = None

def find_landmarks_from_frame(frame):
    
    if debug_frame_path:
        cv2.imwrite(str(debug_frame_path), frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray)

    detections = detector(clahe_image, 1) 

    for k,d in enumerate(detections): 
        landmarks = [] 
        shape = predictor(clahe_image, d) 
        
        for i in range(0,68): 
            landmarks.append((shape.part(i).x, shape.part(i).y))
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) 
            
    if debug_frame_path:
        cv2.imwrite(str(debug_frame_path), frame)
            
    
    return landmarks


def get_normalization_standard_points(landmarks):
    landmarks_array = np.array(landmarks)
    xmax = np.max(landmarks_array[:,0])
    xmin = np.min(landmarks_array[:,0])
    ymax = np.max(landmarks_array[:,1])
    ymin = np.min(landmarks_array[:,1])
    
    return {"xmax":xmax,"xmin":xmin,"ymax":ymax,"ymin":ymin}
    
def normalize_landmarks(landmarks,standard_points):
    normalized_landmarks = []
    x_length =  standard_points['xmax'] -  standard_points['xmin']
    y_length =  standard_points['ymax'] -  standard_points['ymin']
    
    for pair in landmarks:
        normalized_x = (pair[0] - standard_points['xmin']) / float(x_length)
        normalized_y = (pair[1] - standard_points['ymin']) / float(y_length)
        normalized_landmarks.extend((normalized_x,normalized_y))
   
    return normalized_landmarks

def iter_video_files(video_root):
    root = plb.Path(video_root)
    for item in root.iterdir():
        if item.is_file() and item.suffix.lower() == ".mp4":
            yield item
        elif item.is_dir():
            for video_file in item.iterdir():
                if video_file.is_file() and video_file.suffix.lower() == ".mp4":
                    yield video_file


def process_all(data_root=None):
    global debug_frame_path
    pipeline_paths = paths(data_root)
    ensure_dirs(pipeline_paths["exp_labels"])
    debug_frame_path = pipeline_paths["exp_labels"] / "frame.jpg"

    all_landmarks = []
    file_count = 1
    for video_file in iter_video_files(pipeline_paths["video"]):
        print("Preprocess Video " + str(file_count) + ": " + str(video_file))
        file_count += 1
        vidcap = cv2.VideoCapture(str(video_file))
        count = 0
        step = 30
        success = True
        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, count * step)
            success, image = vidcap.read()
            if success:
                count += 1
                landmarks = find_landmarks_from_frame(image)
                if not landmarks:
                    continue
                standard_points = get_normalization_standard_points(landmarks)
                normalized_landmarks = normalize_landmarks(landmarks, standard_points)
                all_landmarks.append(np.array(normalized_landmarks).T.tolist())
        vidcap.release()

    np.save(str(pipeline_paths["exp_labels"] / "generated_landmarks.npy"), np.array(all_landmarks, dtype=np.float32))


if __name__ == "__main__":
    data_root = parse_data_root("Generate normalized facial landmarks from videos")
    process_all(data_root)

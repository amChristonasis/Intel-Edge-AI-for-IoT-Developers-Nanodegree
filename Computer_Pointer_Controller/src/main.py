import cv2
import os
import logging
import numpy as np
from face_detection import FaceDetectionModel
from facial_landmarks_detection import FacialLandmarksDetectionModel
from gaze_estimation import GazeEstimationModel
from head_pose_estimation import HeadPoseEstimationModel
from mouse_controller import MouseController
from argparse import ArgumentParser
from input_feeder import InputFeeder

def build_argparser():
    #Parse command line arguments.

    #:return: command line arguments
    parser = ArgumentParser()
    parser.add_argument("-fdm", "--face_detection_model", required=True, type=str,
                        help="Specify Path to .xml file of Face Detection model.")
    parser.add_argument("-flm", "--facial_landmarks_detection_model", required=True, type=str,
                        help="Specify Path to .xml file of Facial Landmarks Detection model.")
    parser.add_argument("-hpem", "--head_pose_estimation_model", required=True, type=str,
                        help="Specify Path to .xml file of Head Pose Estimation model.")
    parser.add_argument("-gem", "--gaze_estimation_model", required=True, type=str,
                        help="Specify Path to .xml file of Gaze Estimation model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Specify Path to video file or enter cam for webcam")
    parser.add_argument("-flags", "--visualization_flags", required=False, nargs='+',
                        default=[],
                        help="Specify the flags from fd, fld, hp, ge like --flags fd hp fld (Separate each flag by space)"
                             "for seeing the visualization of different model outputs of each frame," 
                             "fd for Face Detection, fld for Facial Landmark Detection"
                             "hp for Head Pose Estimation, ge for Gaze Estimation." )
    parser.add_argument("--cpu_extension", required=False, type=str,
                        default=None,
                        help="CPU Extension for custom layers")
    parser.add_argument("--prob_threshold", required=False, type=float,
                        default=0.6,
                        help="Probability threshold to be used by the model.")
    parser.add_argument("--device", type=str, default="CPU",
                        help="Specify the target device to perform inference on: "
                             "CPU, GPU, FPGA, or MYRIAD")
    
    return parser



def main():

    # Grab command line arguments
    args = build_argparser().parse_args()
    flags = args.visualization_flags
    
    logger = logging.getLogger()
    input_file_path = args.input
    input_feeder = None
    if input_file_path.lower() == "cam":
            input_feeder = InputFeeder("cam")
    else:
        if not os.path.isfile(input_file_path):
            logger.error("Unable to find specified video file")
            exit(1)
        input_feeder = InputFeeder("video", input_file_path)
    
    model_path_dict = {'FaceDetectionModel':args.face_detection_model, 'FacialLandmarksDetectionModel':args.facial_landmarks_detection_model, 
    'GazeEstimationModel':args.gaze_estimation_model, 'HeadPoseEstimationModel':args.head_pose_estimation_model}
    
    for file_name_key in model_path_dict.keys():
        if not os.path.isfile(model_path_dict[file_name_key]):
            logger.error("Unable to find specified " + fileNameKey + " xml file")
            exit(1)
            
    fdm = FaceDetectionModel(model_path_dict['FaceDetectionModel'], args.device, args.cpu_extension)
    fldm = FacialLandmarksDetectionModel(model_path_dict['FacialLandmarksDetectionModel'], args.device, args.cpu_extension)
    gem = GazeEstimationModel(model_path_dict['GazeEstimationModel'], args.device, args.cpu_extension)
    hpem = HeadPoseEstimationModel(model_path_dict['HeadPoseEstimationModel'], args.device, args.cpu_extension)
    
    mc = MouseController('medium','fast')
    
    input_feeder.load_data()
    fdm.load_model()
    fldm.load_model()
    hpem.load_model()
    gem.load_model()
    
    frame_count = 0
    for ret, frame in input_feeder.next_batch():
        if not ret:
            break
        frame_count+=1
        if frame_count%5==0:
            cv2.imshow('video', cv2.resize(frame, (500,500)))
    
        key = cv2.waitKey(60)
        cropped_face, face_coords = fdm.predict(frame, args.prob_threshold)
        if type(cropped_face)==int:
            logger.error("Unable to detect the face.")
            if key==27:
                break
            continue
        
        hp_output = hpem.predict(cropped_face)
        
        left_eye, right_eye, eye_coords = fldm.predict(cropped_face)
        
        new_mouse_coord, gaze_vector = gem.predict(left_eye, right_eye, hp_output)
        
        if (not len(flags)==0):
            preview_frame = frame
            if 'fd' in flags:
                preview_frame = cropped_face
            if 'fld' in flags:
                cv2.rectangle(cropped_face, (eye_coords[0][0]-10, eye_coords[0][1]-10), (eye_coords[0][2]+10, eye_coords[0][3]+10), (0,255,0), 3)
                cv2.rectangle(cropped_face, (eye_coords[1][0]-10, eye_coords[1][1]-10), (eye_coords[1][2]+10, eye_coords[1][3]+10), (0,255,0), 3)
                
            if 'hp' in flags:
                cv2.putText(preview_frame, "Pose Angles: yaw:{:.2f} | pitch:{:.2f} | roll:{:.2f}".format(hp_output[0], hp_output[1], hp_output[2]), (10, 20), cv2.FONT_HERSHEY_COMPLEX, 0.25, (0, 255, 0), 1)
            if 'ge' in flags:
                x, y, w = int(gaze_vector[0]*12), int(gaze_vector[1]*12), 160
                le = cv2.line(left_eye, (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(le, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                re = cv2.line(right_eye, (x-w, y-w), (x+w, y+w), (255,0,255), 2)
                cv2.line(re, (x-w, y+w), (x+w, y-w), (255,0,255), 2)
                cropped_face[eye_coords[0][1]:eye_coords[0][3],eye_coords[0][0]:eye_coords[0][2]] = le
                cropped_face[eye_coords[1][1]:eye_coords[1][3],eye_coords[1][0]:eye_coords[1][2]] = re
                
            cv2.imshow("Visualization", cv2.resize(preview_frame,(500,500)))
        
        if frame_count%5==0:
            mc.move(new_mouse_coord[0], new_mouse_coord[1])    
        if key==27:
                break
    logger.error("VideoStream ended...")
    cv2.destroyAllWindows()
    input_feeder.close()
     
    

if __name__ == '__main__':
    main() 
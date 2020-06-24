import cv2
import numpy as np
from openvino.inference_engine import IECore, IENetwork
import math
'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

class GazeEstimationModel:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.device = device
        self.extensions = extensions
        self.model_structure = self.model_name
        self.model_weights = self.model_name.split(".")[0]+'.bin'
        self.plugin = None
        self.network = None
        self.exec_net = None
        self.input_name = None
        self.input_shape = None
        self.output_names = None
        self.output_shape = None


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        self.plugin = IECore()
        self.network = IENetwork(model=self.model_structure, weights=self.model_weights)
        supported_layers = self.plugin.query_network(network=self.network, device_name=self.device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
        
        
        if len(unsupported_layers)!=0 and self.device=='CPU':
            print(f"Unsupported layers found:{unsupported_layers}")
            if not self.extensions==None:
                print("Adding cpu_extension")
                self.plugin.add_extension(self.extensions, self.device)
                supported_layers = self.plugin.query_network(network = self.network, device_name=self.device)
                unsupported_layers = [l for l in self.network.layers.keys() if l not in supported_layers]
                if len(unsupported_layers)!=0:
                    print("After adding the extension still unsupported layers found")
                    exit(1)
                print("After adding the extension the issue is resolved")
            else:
                print("Provide the path to the cpu extension")
                exit(1)
                
        self.exec_net = self.plugin.load_network(network=self.network, device_name=self.device,num_requests=1)
        
        self.input_name = [i for i in self.network.inputs.keys()]
        self.input_shape = self.network.inputs[self.input_name[1]].shape
        self.output_names = [i for i in self.network.outputs.keys()]


    def predict(self, left_eye_image, right_eye_image, head_pose_angles):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        le_img_processed, re_img_processed = self.preprocess_input(left_eye_image, right_eye_image)
        outputs = self.exec_net.infer({'head_pose_angles':head_pose_angles, 'left_eye_image':le_img_processed, 'right_eye_image':re_img_processed})
        new_mouse_coord, gaze_vector = self.preprocess_output(outputs, head_pose_angles)

        return new_mouse_coord, gaze_vector


    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, left_eye_image, right_eye_image):

        le_image_res = cv2.resize(left_eye_image, (self.input_shape[3], self.input_shape[2]))
        re_image_res = cv2.resize(right_eye_image, (self.input_shape[3], self.input_shape[2]))
        le_img_processed = np.transpose(np.expand_dims(le_image_res, axis=0), (0,3,1,2))
        re_img_processed = np.transpose(np.expand_dims(re_image_res, axis=0), (0,3,1,2))

        return le_img_processed, re_img_processed


    def preprocess_output(self, outputs, head_pose_angles):

        gaze_vector = outputs[self.output_names[0]].tolist()[0]

        rollValue = head_pose_angles[2] #angle_r_fc output from HeadPoseEstimation model
        cosValue = math.cos(rollValue * math.pi / 180.0)
        sinValue = math.sin(rollValue * math.pi / 180.0)
        
        newx = gaze_vector[0] * cosValue + gaze_vector[1] * sinValue
        newy = -gaze_vector[0] *  sinValue + gaze_vector[1] * cosValue
        
        return (newx, newy), gaze_vector

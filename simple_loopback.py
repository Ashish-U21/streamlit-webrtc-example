import cv2
import mediapipe as mp
import math
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

mp_drawing = mp.solutions.drawing_utils
mp_face_detction = mp.solutions.face_detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

class VideoTransformer(VideoTransformerBase):

    def __init__(self):
        self.ls=[]
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        def getAngle(a, b, c):

            ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
            return ang + 360 if ang < 0 else ang
        def distance(a,b):
            dis = math.sqrt( ((a[0]-b[0])**2)+((a[1]-b[1])**2) )
            return dis

        def valid():
            r_ankle = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_hight)
            l_ankle = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_hight)
            l_wrist = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_hight)
            r_knee = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_hight)
            l_knee = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_hight)
            nose = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight)
        #print(image_width,image_hight)
            if r_ankle[1] <= image_hight and l_wrist[0] <= image_width and nose[1]>10 and l_ankle[1] <= image_hight:
                
                return True
            return False 
        
        #count=0
        c=0
        s_count = 0
        s_c = 0
        j_count = 0
        j_c = 0
        prev=""
        prev1=""
        pre2=""
        pre3=""
        pre=""
        jumping_jack_counter=0
        frame_ct = 0
        jumping_frame_counter = 0
        jumping_frame_prev = 0 
        c_position = ""
        p_position = ""


        
            
        frame_ct = frame_ct+1
        #print("***********************",frame_ct)
        image_hight, image_width, _ = image.shape
        #print(image)
        image.flags.writeable = False 
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_RGB2BGR)
        #image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        #image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        #image = cv2.resize(image,(480,600), fx = 0, fy = 0,interpolation=cv2.INTER_CUBIC)
        pose_results = pose.process(image)
        mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        try:

            nose = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_hight)
            r_wrist = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image_hight)
            l_wrist = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image_hight)
            r_elbow = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image_hight)
            l_elbow = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image_hight)
            r_shoulder = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_hight)
            l_shoulder = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_hight)
            l_hip = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_hight)
            r_hip = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_hight)
            r_knee = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image_hight)
            l_knee = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image_hight)
            r_ankle = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image_hight)
            l_ankle = (pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image_width , pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image_hight)

            ankle_dist = distance(l_ankle,r_ankle)
            knee_angle = int(getAngle(l_ankle,l_knee,l_hip))
            shoulder_dist = distance(l_shoulder,r_shoulder)
            hip_dist = distance(l_hip,r_hip)
            r_lower = distance(r_ankle,r_hip)
            l_lower = distance(l_ankle,l_hip)
            r_hip_dist = distance(r_hip,r_shoulder)
            l_hip_dist = distance(l_hip,l_shoulder)
            r_knee_dist = distance(r_knee,r_hip)
            l_knee_dist = distance(l_knee,l_hip)

            knee_dist = distance(l_knee,r_knee)
            left_knee_angle = int(getAngle(l_ankle,l_knee,l_hip))
            right_knee_angle = int(getAngle(r_hip,r_knee,r_ankle))
            r_diff = int(r_lower-r_hip_dist)
            l_diff = int(l_lower-l_hip_dist)
            left_s_h_k_angle = int(getAngle(l_shoulder,l_hip,l_knee))
            right_s_h_k_angle = int(getAngle(r_knee,r_hip,r_shoulder))
            #print(r_ankle[1],image_hight)

            if valid()==True:
                
                #print('==================================================================================')
                if (l_wrist[1] > l_shoulder[1]) and (r_wrist[1] > r_shoulder[1]) and (ankle_dist < shoulder_dist):
                    pre1 = l_wrist[1]
                    pre2 = l_shoulder[1]
                    pre3 = r_wrist[1]
                    pre4 = r_shoulder[1]
                    pre5 = ankle_dist
                    pre6 = shoulder_dist
                    if l_wrist[1]==pre1 and l_shoulder[1]==pre2 and r_wrist[1]==pre3 and r_shoulder[1]==pre4 and ankle_dist==pre5 and shoulder_dist==pre6:
                        print("==================================Start===========")
                        c_position = "Start"
                        self.ls.append(c_position)
                        j_c = 0
                        j_count = j_count+1
                    
                elif (l_wrist[1] < l_shoulder[1]) and (r_wrist[1] < r_shoulder[1]) and (ankle_dist > shoulder_dist) and (knee_angle in range(170,190)):
                    pre8 = l_wrist[1]
                    pre9 = l_shoulder[1]
                    pre10 = r_wrist[1]
                    pre11 = r_shoulder[1]
                    pre12 = ankle_dist
                    pre13 = shoulder_dist
                    if l_wrist[1]==pre8 and l_shoulder[1]==pre9 and r_wrist[1]==pre10 and r_shoulder[1]==pre11 and ankle_dist==pre12 and shoulder_dist==pre13:
                        print("*************************End*********************")
                        c_position = "End"
                        self.ls.append(c_position)
                        j_c = j_c+1
                        j_count = 0
                #print("==================",frame_ct) 

                
                c=""
                p=""
                count=0
                
                for i in self.ls:
                    
                    c=i
                    
                    
                    if p=="Start" and c=="End":
                        
                        count=count+1
                        

                    p=i 

                    
                
                image = cv2.putText(image,'JumpingJack %i' %(count), (15,50), cv2.FONT_HERSHEY_SIMPLEX, 
                           2, (55,140,55), 5, cv2.LINE_AA)  
                            

                if j_count==1 and  (l_wrist[1] > l_shoulder[1]) and (r_wrist[1] > r_shoulder[1]) and (ankle_dist < shoulder_dist):
                    jumping_jack_counter=jumping_jack_counter+1
                    jumping_frame_prev = jumping_frame_counter
                    jumping_frame_counter = 0
                    print("====================",jumping_jack_counter)
                if jumping_jack_counter > 1:
                    
                    image = cv2.putText(image,'JumpingJack %i' %(jumping_jack_counter-1), (15,50), cv2.FONT_HERSHEY_SIMPLEX, 
                           2, (55,140,55), 5, cv2.LINE_AA)

                    #speed  = fps/jumping_frame_prev
                    #jumping_speed = format(speed,".1f")

                    #print("=========Speed========",speed)
                    """
                    image = cv2.putText(image, 'JumpingJack %s/second' %jumping_speed, (15,90), cv2.FONT_HERSHEY_SIMPLEX, 
                                1, (75,85,240), 2, cv2.LINE_AA)    
                    """  

        except Exception as e:
                print(e)
                #print("Not in the frame")
                #raise



        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        #print(image)
        return image

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
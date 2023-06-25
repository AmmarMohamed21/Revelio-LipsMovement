import cv2
import os
import dlib
import numpy as np
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

DATADIR = "dataset"
REALDATASETDIR = DATADIR + "/real"
FAKEDATASETDIR_df = DATADIR + "/deepfakes"
FAKEDATASETDIR_f2f = DATADIR + "/face2face"
FAKEDATASETDIR_fs = DATADIR + "/faceswap"
FAKEDATASETDIR_nt = DATADIR + "/neuraltextures"

REALDATASETVIDEOS = [] 
FAKEDATASETVIDEOS_df = [] 
FAKEDATASETVIDEOS_f2f = []
FAKEDATASETVIDEOS_fs = [] 
FAKEDATASETVIDEOS_nt = []

REALDATASETVIDEOS = [] 
FAKEDATASETVIDEOS_df = [] 
FAKEDATASETVIDEOS_f2f = []
FAKEDATASETVIDEOS_fs = [] 
FAKEDATASETVIDEOS_nt = []

MOUTHSTARTINDEX = 48
MOUTHENDINDEX = 68

LandmarksPredictorPath = "landmarkPredictor/shape_predictor_68_face_landmarks.dat"

class Preprocessor:
    def __init__(self, dataset=None, datasetType=None, targetShape=(96,96), targetPath="mouthsDataset", useDlib = True, isCropping=False):
        if isCropping:
            self.dataset = dataset
            self.datasetType = datasetType
            self.targetShape = targetShape
            self.targetPath = targetPath
            if useDlib:
                self.faceDetector = dlib.get_frontal_face_detector()
                self.faceLandmarksPredictor = dlib.shape_predictor(LandmarksPredictorPath)
            else:
                self.faceDetector = None #TODO
                self.faceLandmarksPredictor = None
            
            REALDATASETVIDEOS = os.listdir(REALDATASETDIR)
            FAKEDATASETVIDEOS_df = os.listdir(FAKEDATASETDIR_df)
            FAKEDATASETVIDEOS_f2f = os.listdir(FAKEDATASETDIR_f2f)
            FAKEDATASETVIDEOS_fs = os.listdir(FAKEDATASETDIR_fs)
            FAKEDATASETVIDEOS_nt = os.listdir(FAKEDATASETDIR_nt)

            REALDATASETVIDEOS = [REALDATASETDIR + "/" + video for video in REALDATASETVIDEOS]
            FAKEDATASETVIDEOS_df = [FAKEDATASETDIR_df + "/" + video for video in FAKEDATASETVIDEOS_df]
            FAKEDATASETVIDEOS_f2f = [FAKEDATASETDIR_f2f + "/" + video for video in FAKEDATASETVIDEOS_f2f]
            FAKEDATASETVIDEOS_fs = [FAKEDATASETDIR_fs + "/" + video for video in FAKEDATASETVIDEOS_fs]
            FAKEDATASETVIDEOS_nt = [FAKEDATASETDIR_nt + "/" + video for video in FAKEDATASETVIDEOS_nt]

    


    #read video frames
    def readVideoFrames(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()
        return frames
    
    def detectMouthLandmarksFromImage(self, img, cachedFace = None, useCachedFace=False):
        #Detect facial landmarks using dlib
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #get mouth landmarks
        mouth_landmarks, face = self.getMouthLandmarks(gray, cachedFace, useCachedFace)

        #Get bounding box of mouth region
        (x, y, w, h) = cv2.boundingRect(np.array(mouth_landmarks))

        #middle point of mouth region
        (x, y) = (int((x + x + w) / 2), int((y + y + h) / 2))

        #Make the box of size twice the width
        w = w * 2
        h = w 
        #update x and y
        x = x - w // 2
        y = y - h // 2

        #Crop the mouth region
        mouth = gray[y:y+h, x:x+w]

        #resize mouth shape
        mouth = cv2.resize(mouth, self.targetShape)
        
        return mouth, face


    def getMouthLandmarks(self, gray, cachedFace = None, useCachedFace=False):
        
        # # Detect faces in the grayscale image
        detectedFaces = None if useCachedFace else self.faceDetector(gray)
        if detectedFaces is None or len(detectedFaces) == 0:
            useCachedFace = True
        face = cachedFace if useCachedFace else detectedFaces[0]

        #If no face is detected use whole image as face
        if face is None:
            face = dlib.rectangle(0, 0, gray.shape[1], gray.shape[0])

        # Loop over each detected face
        # for face in faces:
            # Detect landmarks for the current face
        landmarks = self.faceLandmarksPredictor(gray, face)

        mouth_region = landmarks.parts()[MOUTHSTARTINDEX:MOUTHENDINDEX]
        mouth_landmarks = []
        for p in mouth_region:
            mouth_landmarks.append([p.x, p.y])
        
        return mouth_landmarks, face

    def processVideo(self, video):
        cachedFace = None
        useCachedFace = False

        #create directory for the video
        if not os.path.exists(self.targetPath + "/" + video.split("/")[-1].split(".")[0]):
            os.makedirs(self.targetPath + "/" + video.split("/")[-1].split(".")[0])
        # else:
        
        frames = self.readVideoFrames(video)
        if len(os.listdir(self.targetPath + "/" + video.split("/")[-1].split(".")[0])) == len(frames):
            return
        for i, frame in enumerate(frames):
            #If image exists continue
            if os.path.exists(self.targetPath + "/" + video.split("/")[-1].split(".")[0] + "/" + str(i).zfill(4) + ".png"):
                continue
            #Detect Face once each 5 frames
            useCachedFace = (True if i % 5 != 0 else False) and cachedFace is not None
            mouth, cachedFace = self.detectMouthLandmarksFromImage(frame, cachedFace, useCachedFace)
            #save mouth image
            imageName = str(i).zfill(4)
            cv2.imwrite(self.targetPath + "/" + video.split("/")[-1].split(".")[0] + "/" + imageName + ".png", mouth)
        
    def prepareMouthsDataset(self): #, noFrames=25):

        #Create target directory
        if not os.path.exists(self.targetPath):
            os.makedirs(self.targetPath)
        
        if not os.path.exists(self.targetPath + "/" + self.datasetType):
            os.makedirs(self.targetPath + "/" + self.datasetType)

        self.targetPath = self.targetPath + "/" + self.datasetType
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()-6) as pool:
            pool.map(self.processVideo, self.dataset)


    def framesToSequences(self, args):
        folder_path, vid, max_frames, sequence_length = args
        print(vid)
        vid_frames = os.listdir(os.path.join(folder_path, vid))
        #split in a sequence of 25 frames, discard the last frames if not enough
        vid_frames = vid_frames[:min(max_frames, len(vid_frames)-len(vid_frames)%sequence_length)]
        vid_frames = [vid_frames[i:i+sequence_length] for i in range(0, len(vid_frames), sequence_length)]
        final_sequences = []
        for sequence in vid_frames:
            try:
                final_sequences.append(np.array([np.array(Image.open(os.path.join(folder_path+vid, filename))) for filename in sequence]))
            except:
                continue
            
        #vid_frames = [np.array([np.array(Image.open(os.path.join(folder_path+vid, filename))) for filename in vid_frame_sequence]) for vid_frame_sequence in vid_frames]
        return final_sequences
    

    def saveDatasetAsNp(self, datasetPath, outputName, sequenceLength=25, maxFrames=300):
        dataset = []

        with ThreadPoolExecutor(max_workers=12) as executor:
            # Submit a task for each video path
            tasks = [executor.submit(self.framesToSequences, (datasetPath, vid_path, maxFrames, sequenceLength)) for vid_path in os.listdir(datasetPath)]

            # Wait for all tasks to finish and get their results
            for task in tasks:
                dataset.extend(task.result())

        dataset = np.array(dataset)
        np.savez_compressed(outputName, dataset=dataset)

    

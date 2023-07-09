from models.resnet_feature_extractor import *
import numpy as np

def cropAndNormalizeFrames(frames, targetShape=(88,88), mean=127.231, std=42.438):
    '''
    A Function that is used to crop and normalize the frames of a video to prepare it for the feature extractor

    Parameters:
    frames (numpy array): The frames of the video
    targetShape: The shape of the frames after cropping
    mean: The mean of the dataset (default is the mean of the FF++ dataset)
    std: The standard deviation of the dataset (default is the standard deviation of the FF++ dataset)
    ''' 
    frames = frames.astype(np.float32)
    L,W,H = frames.shape
    x_start = (W - targetShape[0]) // 2
    y_start = (H - targetShape[1]) // 2
    x_end = x_start + targetShape[0]
    y_end = y_start + targetShape[1]

    normalizedFrames = (frames - mean) / std
    croppedFrames = normalizedFrames[:, x_start:x_end, y_start:y_end]
    return croppedFrames

def cropAndNormalizeDataset(dataPath, targetShape=(88,88), mean=127.231, std=42.438):
    '''
    A Function that is used to crop and normalize the frames of the full training dataset to prepare it for the feature extractor

    Parameters:
    dataPath: The path of the dataset
    targetShape: The shape of the frames after cropping
    mean: The mean of the dataset (default is the mean of the FF++ dataset)
    std: The standard deviation of the dataset (default is the standard deviation of the FF++ dataset)
    '''
    data = np.load(dataPath, mmap_mode='r')['dataset'] 
    L, S, W, H = data.shape
    x_start = (W - targetShape[0]) // 2
    y_start = (H - targetShape[1]) // 2
    x_end = x_start + targetShape[0]
    y_end = y_start + targetShape[1]

    chunk_size = 1000

    croppedData = np.zeros((L, S, targetShape[0], targetShape[1]), dtype=np.float32)
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:min(i+chunk_size, len(data))].astype(np.float32)
        normalized_chunk = (chunk - mean) / std
        croppedData[i:min(i+chunk_size, len(data))] = normalized_chunk[:, :, x_start:x_end, y_start:y_end]
    
    return croppedData

def ExtractFeaturesFromMouthFramesDataset(dataPath, targetPath, resnetModelPath = "pretrainedModels/resnet_feature_extractor.pth"):
    '''
    A Function that is used to extract the features of the mouth frames of the full training dataset

    Parameters:
    dataPath: The path of the dataset
    targetPath: The path of the file where the features will be saved
    resnetModelPath: The path of the pretrained feature extractor model
    '''

    data = cropAndNormalizeDataset(dataPath)
    featureExtractor = load_resnet_feature_extractor(resnetModelPath)
    featureExtractor.eval()
    features = ExtractFeatures(featureExtractor, data)
    np.savez_compressed(targetPath, features=features)

def ExtractFeatures(featureExtractor, data):
    '''
    A Function that is used to extract the features of the mouth frames of a video
    
    Parameters:
    featureExtractor: The pretrained ResNet18 feature extractor model objet
    data: The mouth frames of the video
    '''
    features = np.array([featureExtractor(torch.FloatTensor(data[i][None, None, :, :, :]).cuda(), lengths=None).cpu().detach().numpy() for i in range(data.shape[0])])
    features=features.reshape(features.shape[0], features.shape[2], features.shape[3])
    return features




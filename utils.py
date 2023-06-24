import numpy as np

def getMeanAndStd(realData, deepfakesData, face2faceData, faceswapData, neuraltexturesData):  
    realmean = realData.mean()
    realSize = realData.shape[0]

    deepfakesmean = deepfakesData.mean()
    deepfakesSize = deepfakesData.shape[0]

    face2facemean = face2faceData.mean()
    face2faceSize = face2faceData.shape[0]

    faceswapmean = faceswapData.mean()
    faceswapSize = faceswapData.shape[0]

    neuraltexturesmean = neuraltexturesData.mean()
    neuraltexturesSize = neuraltexturesData.shape[0]

    datasetmean = (realmean * realSize + deepfakesmean * deepfakesSize + face2facemean * face2faceSize + faceswapmean * faceswapSize + neuraltexturesmean * neuraltexturesSize) / (realSize + deepfakesSize + face2faceSize + faceswapSize + neuraltexturesSize)

    chunks = np.array_split(realData, len(realData) // 1000 + 1)
    sum_sq_diffs = 0
    for chunk in chunks:
        sq_diffs = np.square(chunk - datasetmean)
        sum_sq_diffs += np.sum(sq_diffs)

    chunks = np.array_split(deepfakesData, len(deepfakesData) // 1000 + 1)
    for chunk in chunks:
        sq_diffs = np.square(chunk - datasetmean)
        sum_sq_diffs += np.sum(sq_diffs)

    chunks = np.array_split(face2faceData, len(face2faceData) // 1000 + 1)
    for chunk in chunks:
        sq_diffs = np.square(chunk - datasetmean)
        sum_sq_diffs += np.sum(sq_diffs)

    chunks = np.array_split(faceswapData, len(faceswapData) // 1000 + 1)
    for chunk in chunks:
        sq_diffs = np.square(chunk - datasetmean)
        sum_sq_diffs += np.sum(sq_diffs)

    chunks = np.array_split(neuraltexturesData, len(neuraltexturesData) // 1000 + 1)
    for chunk in chunks:
        sq_diffs = np.square(chunk - datasetmean)
        sum_sq_diffs += np.sum(sq_diffs)

    datasetstd = np.sqrt(sum_sq_diffs / ((realSize + deepfakesSize + face2faceSize + faceswapSize + neuraltexturesSize)*25*96*96))
    
    return datasetmean, datasetstd

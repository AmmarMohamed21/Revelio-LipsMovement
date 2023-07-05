import numpy as np
import os
from preprocessor import *
import torch
from models.mstcn import *
from models.resnet_feature_extractor import *
from FeatureExtractor import *
import time
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score

class LipMovementClassifier:
    def __init__(self, isPredictor=False, predictionMSTCNModelPaths = [], pretrainedMSTCNModelPath='pretrainedModels/mstcn.pth', mstcnConfigFilePath='models/configs/mstcn.json', featureExtractorModelPath=None, fake_features_path=None, learning_rate=2e-4, batch_size=32, epochs=10):
        assert mstcnConfigFilePath is not None
        self.configFile = mstcnConfigFilePath

        if isPredictor:
            assert featureExtractorModelPath is not None
            assert len(predictionMSTCNModelPaths) > 0
            self.predictionMSTCNModels = []
            for modelPath in predictionMSTCNModelPaths:
                self.predictionMSTCNModels.append(load_mstcn_model(modelPath,configfile=self.configFile))
            self.featureExtractor = load_resnet_feature_extractor(featureExtractorModelPath)
            self.featureExtractor.eval()


        else:
            assert pretrainedMSTCNModelPath is not None
            assert fake_features_path is not None
            self.pretrainedMSTCNModelPath = pretrainedMSTCNModelPath
            self.MSTCNModel = load_mstcn_model(self.pretrainedMSTCNModelPath,configfile=self.configFile)
            self.OutputDir = "trainedclassifiers/model"+str(len(os.listdir("trainedclassifiers"))+1)
            os.makedirs(self.OutputDir)

            self.report = open(self.OutputDir+"/report.txt", "w")

            self.report.write("Pretrained Model Path: "+self.mstcnModelPath+"\n")
            self.report.write("Config File: "+self.configFile+"\n")

            self.MSTCNModel = load_mstcn_model(self.pretrainedModelPath,configfile=self.configFile)

            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.epochs = epochs

            self.FakeFeaturesPath = fake_features_path
            self.RealFeaturesPath = 'featuresExtracted/realfeatures.npz'

            self.report.write("Learning Rate: "+str(self.learning_rate)+"\n")
            self.report.write("Batch Size: "+str(self.batch_size)+"\n")
            self.report.write("Fake Features Path: "+self.FakeFeaturesPath+"\n")
            self.report.write("Real Features Path: "+self.RealFeaturesPath+"\n")
            
            fakeFeatures = np.load(self.FakeFeaturesPath)['features']
            realFeatures = np.load(self.RealFeaturesPath)['features']

            #Split last 2400 frames (last 140 videos) as test set
            fakeTestFeatures = fakeFeatures[-1680:]
            realTestFeatures = realFeatures[-1680:]
            fakeTrainFeatures = fakeFeatures[:-1680]
            realTrainFeatures = realFeatures[:-1680]

            #Labels
            fakeTestLabels = np.zeros((fakeTestFeatures.shape[0],1))
            realTestLabels = np.ones((realTestFeatures.shape[0],1))

            fakeTrainLabels = np.zeros((fakeTrainFeatures.shape[0],1))
            realTrainLabels = np.ones((realTrainFeatures.shape[0],1))

            #Concatenate datasets
            self.trainFeatures = np.concatenate((fakeTrainFeatures, realTrainFeatures), axis=0)
            self.trainLabels = np.concatenate((fakeTrainLabels, realTrainLabels), axis=0)
            self.testFeatures = np.concatenate((fakeTestFeatures, realTestFeatures), axis=0)
            self.testLabels = np.concatenate((fakeTestLabels, realTestLabels), axis=0)

            self.trainloader = torch.utils.data.DataLoader(list(zip(self.trainFeatures, self.trainLabels)), batch_size=self.batch_size, shuffle=True)
            self.testloader = torch.utils.data.DataLoader(list(zip(self.testFeatures, self.testLabels)), batch_size=self.batch_size)
    
    def predict(self, inputGrayFrames, inputLandmarks):
        croppedMouths = np.array([Preprocessor().cropMouthFromImage(inputGrayFrames[i], inputLandmarks[i]) for i in range(len(inputGrayFrames))])
        croppedMouths = cropAndNormalizeFrames(croppedMouths)
        
        #split into sequences each of size 25 (L,W,H) to (L/25,25,W,H)
        croppedMouths = np.array([croppedMouths[i:i+25] for i in range(0, len(croppedMouths), 25)])

        #extract features
        features = ExtractFeatures(self.featureExtractor, croppedMouths)

        #convert to tensor
        features = torch.from_numpy(features).cuda().float()

        #predict
        all_predictions = []
        for model in self.predictionMSTCNModels:
            predictions = model(features, lengths=[features.shape[1] for i in range(features.shape[0])])
            all_predictions.append(predictions.detach().cpu().numpy())

        return all_predictions
    
    def train(self):
        optimizer = torch.optim.Adam(self.MSTCNModel.parameters(), lr=self.learning_rate)
        criterion = nn.BCELoss()

        startTime = time.time()

        for epoch in range(self.epochs):
            running_loss = 0.0
            all_labels = []
            all_preds = []
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs = inputs.cuda().float()
                labels = labels.cuda().float()
                optimizer.zero_grad()
                outputs = self.MSTCNModel(inputs, lengths=[inputs.shape[1] for i in range(inputs.shape[0])])
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(labels.detach().cpu().numpy())
                

            epochAccuracy = self.calculateAccuracy(all_preds, all_labels)
            epochAUC = self.calculateAUC(all_preds, all_labels)
            print('Epoch: %d, loss: %.3f' % (epoch + 1, running_loss / (i + 1)))
            self.report.write('Epoch: %d, loss: %.3f' % (epoch + 1, running_loss / (i + 1))+"\n")
            print('accuracy: %.3f' % (epochAccuracy))
            self.report.write('accuracy: %.3f' % (epochAccuracy)+"\n")
            print('AUC: %.3f' % (epochAUC))
            self.report.write('AUC: %.3f' % (epochAUC)+"\n")

        endTime = time.time()

        #calculate time taken
        timeTaken = endTime - startTime
        print('Time of Training: %.3f' % (timeTaken))
        self.report.write('Time of Training: %.3f' % (timeTaken)+"\n")
    
    def test(self):
        self.MSTCNModel.eval()

        #Calculate accuracy on test set
        all_labels = []
        all_preds = []
        for i, data in enumerate(self.testloader, 0):
            inputs, labels = data
            inputs = inputs.cuda().float()
            labels = labels.cuda().float()
            outputs = self.MSTCNModel(inputs, lengths=[inputs.shape[1] for i in range(inputs.shape[0])])
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

        print('Test Set Evaluation:')
        testVideosAccuracy = self.calculateAccuracyByVideos(all_preds, all_labels)
        testVideosAUC = self.calculateAUCByVideos(all_preds, all_labels)
        testVideoCM, testVideoReport = self.calculateCMAndReportByVideo(all_preds, all_labels)

        testAccuracy = self.calculateAccuracy(all_preds, all_labels)
        testAUC = self.calculateAUC(all_preds, all_labels)
        testCM, testReport = self.calculateCMAndReport(all_preds, all_labels)

        print('Test Videos Accuracy: %.3f' % (testVideosAccuracy))
        self.report.write('Test Videos Accuracy: %.3f' % (testVideosAccuracy)+"\n")
        print('Test Videos AUC: %.3f' % (testVideosAUC))
        self.report.write('Test Videos AUC: %.3f' % (testVideosAUC)+"\n")
        print('Test Videos Confusion Matrix:')
        self.report.write('Test Videos Confusion Matrix:'+"\n")
        print(testVideoCM)
        self.report.write(str(testVideoCM)+"\n")
        print('Test Videos Classification Report:')
        self.report.write('Test Videos Classification Report:'+"\n")
        print(testVideoReport)
        self.report.write(str(testVideoReport)+"\n")

        print('Test Accuracy: %.3f' % (testAccuracy))
        self.report.write('Test Accuracy: %.3f' % (testAccuracy)+"\n")
        print('Test AUC: %.3f' % (testAUC))
        self.report.write('Test AUC: %.3f' % (testAUC)+"\n")
        print('Test Confusion Matrix:')
        self.report.write('Test Confusion Matrix:'+"\n")
        print(testCM)
        self.report.write(str(testCM)+"\n")
        print('Test Classification Report:')
        self.report.write('Test Classification Report:'+"\n")
        print(testReport)
        self.report.write(str(testReport)+"\n")


        torch.save(self.MSTCNModel.state_dict(), self.OutputDir+'/lips_movements_classifer.pth')
        self.report.close()
            
    def calculateAccuracy(self,preds,labels):
        preds = np.array(preds).round().flatten()
        labels = np.array(labels).flatten()
        return (preds == labels).mean()
    
    def calculateAUC(self, preds,labels):
        preds = np.array(preds).flatten()
        labels = np.array(labels).flatten()
        return roc_auc_score(labels,preds)
    
    def calculateAccuracyByVideos(self, preds,labels):
        preds = np.array(preds).round().flatten()
        labels = np.array(labels).flatten()
        preds = preds.reshape(-1,12)
        labels = labels.reshape(-1,12)
        preds = np.mean(preds,axis=1).round()
        labels = np.mean(labels,axis=1).round()
        return (preds == labels).mean()
    
    def calculateAUCByVideos(self,preds,labels):
        preds = np.array(preds).flatten()
        labels = np.array(labels).flatten()
        preds = preds.reshape(-1,12)
        labels = labels.reshape(-1,12)
        preds = np.mean(preds,axis=1)
        labels = np.mean(labels,axis=1)
        return roc_auc_score(labels,preds)
    
    def calculateCMAndReportByVideo(self, preds,labels):
        preds = np.array(preds).round().flatten()
        labels = np.array(labels).flatten()
        preds = preds.reshape(-1,12)
        labels = labels.reshape(-1,12)
        preds = np.mean(preds,axis=1).round()
        labels = np.mean(labels,axis=1).round()
        cm = confusion_matrix(labels,preds)
        report = classification_report(labels,preds)
        return cm,report
    
    def calculateCMAndReport(self, preds,labels):
        preds = np.array(preds).round().flatten()
        labels = np.array(labels).flatten()
        return confusion_matrix(labels,preds), classification_report(labels,preds)
    



    
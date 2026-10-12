"""FBSAANNpt_FBSAANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2026 BAI Research (bairesearch.com.au)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
FBSAANNpt Forward Backward Self-Assembly artificial neural network (FBSAANN)

"""


import torch as pt

from ANNpt_globalDefs import *
import FBSAANNpt_FBSAANNmodel
import ANNpt_data

def createModel(dataset):
	datasetSize = ANNpt_data.getDatasetSize(dataset, printSize=True)
	numberOfFeatures = ANNpt_data.countNumberFeatures(dataset)
	numberOfClasses, numberOfClassSamples = ANNpt_data.countNumberClasses(dataset)
	
	print("creating new model")
	if(printFBSAANNmodelProperties):
		print("Creating new model:")
		print("\t ---")
		print("\t datasetType = ", datasetType)
		print("\t stateTrainDataset = ", stateTrainDataset)
		print("\t stateTestDataset = ", stateTestDataset)
		print("\t ---")
		print("\t datasetName = ", datasetName)
		print("\t datasetRepeatSize = ", datasetRepeatSize)
		print("\t trainNumberOfEpochs = ", trainNumberOfEpochs)
		print("\t ---")
		print("\t batchSize = ", batchSize)
		print("\t numberOfLayers = ", numberOfLayers)
		print("\t hiddenLayerSize = ", hiddenLayerSize)
		print("\t inputLayerSize (numberOfFeatures) = ", numberOfFeatures)
		print("\t outputLayerSize (numberOfClasses) = ", numberOfClasses)
		print("\t ---")
		print("\t trainLocal = ", trainLocal)
		print("\t useFBSA = ", useFBSA)
		if(useFBSA):
			print("\t fbsaBackwardWidthMultiplier = ", fbsaBackwardWidthMultiplier)
			print("\t fbsaLearningRateFactors = ", fbsaLearningRateFactors)
			print("\t fbsaWeightDecay = ", fbsaWeightDecay)
		print("\t ---")
		print("\t useImageDataset = ", useImageDataset)
		if(useImageDataset):
			print("\t\t\t numberOfConvlayers = ", numberOfConvlayers)
			print("\t\t\t numberOfFFLayers = ", numberOfFFLayers)

	config = FBSAANNpt_FBSAANNmodel.FBSAANNconfig(
		batchSize = batchSize,
		numberOfLayers = numberOfLayers,
		numberOfConvlayers = numberOfConvlayers,
		hiddenLayerSize = hiddenLayerSize,
		CNNhiddenLayerSize = CNNhiddenLayerSize,
		inputLayerSize = numberOfFeatures,
		outputLayerSize = numberOfClasses,
		linearSublayersNumber = linearSublayersNumber,
		numberOfFeatures = numberOfFeatures,
		numberOfClasses = numberOfClasses,
		datasetSize = datasetSize,
		numberOfClassSamples = numberOfClassSamples
	)
	model = FBSAANNpt_FBSAANNmodel.FBSAANNmodel(config)
	
	print(model)
	#summary(model, input_size=(3, 32, 32))  # adjust input_size as needed

	return model

def createOptimiser(model):
	"""Apply the local SAL directions with plain SGD, including the shared scheduler.

	The model supplies analytical local correlations as negative gradients. Adam or
	momentum would change the paper's plasticity dynamics. Learning-rate factors
	scale correlations only; weight decay is uniform across all four families.
	"""
	if(not model.useFBSA):
		raise ValueError("The FBSA optimiser requires useFBSA=True")
	return pt.optim.SGD(model.parameters(), lr=learningRate, weight_decay=model.config.fbsaWeightDecay)
	
def trainOrTestModel(model, trainOrTest, x, y, optim, l):
	loss, accuracy = model(trainOrTest, x, y, optim, l)
	return loss, accuracy




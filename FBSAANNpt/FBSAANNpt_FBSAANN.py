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
			print("\t fbsaImplementation = ", fbsaImplementation)
			print("\t fbsaBackwardWidthMultiplier = ", fbsaBackwardWidthMultiplier)
			print("\t fbsaLearningRateFactors = ", fbsaLearningRateFactors)
			print("\t fbsaMomentum = ", fbsaMomentum)
			print("\t fbsaWeightDecay = ", fbsaWeightDecay)
			print("\t fbsaWeightDecayV = ", fbsaWeightDecayV)
			print("\t fbsaWeightDecayVBar = ", fbsaWeightDecayVBar)
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
	"""PNAS 2026 SGD: local gradient scaling, then family-specific decay/momentum.

	One optimiser is equivalent to the reference's two simultaneous SGD steps.
	Keep the ANNpt wrapper's optimiser/scheduler lifetime across epochs. Factors
	belong on gradients, not group learning rates (which would also scale decay).
	"""
	if(not model.useFBSA):
		raise ValueError("The FBSA optimiser requires useFBSA=True")
	config = model.config
	groups = []
	for name, layers, decay in (
		("forward", model.layersLinearF, config.fbsaWeightDecay),
		("backward", model.layersLinearB, config.fbsaWeightDecay),
		("lateralForward", model.layersLateralF, config.fbsaWeightDecayV),
		("lateralBackward", model.layersLateralB, config.fbsaWeightDecayVBar),
	):
		parameters = [parameter for parameter in layers.parameters() if parameter.requires_grad]
		if(parameters):
			groups.append({"params": parameters, "weight_decay": decay, "name": name})
	return pt.optim.SGD(groups, lr=learningRate, momentum=config.fbsaMomentum)
	
def trainOrTestModel(model, trainOrTest, x, y, optim, l):
	loss, accuracy = model(trainOrTest, x, y, optim, l)
	return loss, accuracy




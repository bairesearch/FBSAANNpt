"""FBSAANNpt_FBSAANN_globalDefs.py

# Author:
Richard Bruce Baxter - Copyright (c) 2026 BAI Research (bairesearch.com.au)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
FBSAANNpt FBSAANN globalDefs

"""

#debug parameters:
printFBSAANNmodelProperties = True

#dataset parameters:
useTabularDataset = True
useImageDataset = False	#use CIFAR-10 dataset with CNN 

#FBSAANN architecture parameters:
useFBSA = True	#default: True
if(useFBSA):
	trainLocal = True
else:
	trainLocal = False	#default: False #disabled for debug/benchmark against standard full layer backprop
	#standard backprop - backward connections not used

#Self-Assembling Learning (SAL), Liao et al. (2024), equations (2), (5)-(8).
#These are tunable defaults, not a reproduction of the paper's hyperparameter search.
fbsaBackwardWidthMultiplier = 1.0	#hidden feedback width / forward width; input/output widths are unchanged
fbsaLearningRateFactors = {"forward": 1.0, "backward": 1.0, "lateralForward": 0.1, "lateralBackward": 1.0}
fbsaWeightDecay = 1e-4	#uniform gamma: delta = learningRate * (factor * correlation - gamma * weight)
#The base learningRate, batch size, architecture and scheduler still come from ANNpt_globalDefs.
#D_d = I: signed errors propagate linearly; do not apply the forward ReLU derivative.
activationFunctionTypeBackward = "none"


#skip layer parameters:
supportSkipLayers = False #default: False 	#fully connected skip layer network
supportSkipLayersResidual = False	#default: False	#direct residual connections

if(useImageDataset):
	useCNNlayers = True		#mandatory:True
elif(useTabularDataset):
	useCNNlayers = False	 #default:False	#optional	#enforce different connection sparsity across layers to learn unique features with greedy training	#use 2D CNN instead of linear layers

#CNN parameters:
if(useImageDataset):
	#create CNN architecture, where network size converges by a factor of ~4 (or 2*2) per layer and number of channels increases by the same factor
	CNNkernelSize = 3
	CNNstride = 1
	CNNpadding = "same"
	useCNNlayers2D = True
	CNNinputWidthDivisor = 2
	CNNinputSpaceDivisor = CNNinputWidthDivisor*CNNinputWidthDivisor
	CNNinputPadding = False
	CNNmaxInputPadding = False	#pad input with zeros such that CNN is applied to every layer
	debugCNN = False
	if(CNNstride == 1):
		CNNmaxPool = True
		#assert not supportSkipLayers, "supportSkipLayers not currently supported with CNNstride == 1 and CNNmaxPool"
	elif(CNNstride == 2):
		CNNmaxPool = False
		assert CNNkernelSize==2
	else:
		print("error: CNNstride>2 not currently supported")
	CNNbatchNorm = True
else:
	CNNmaxPool = False
	CNNbatchNorm = False

trainNumberOfEpochsLow = False	#default: False
trainNumberOfEpochsHigh = False	#default: False	#orig: True	#use ~4x more epochs to train

#activation function parameters:
activationFunctionTypeForward = "relu"

#loss function parameters:
useInbuiltCrossEntropyLossFunction = True	#required

#sublayer parameters:	
simulatedDendriticBranches = False	#optional	#performTopK selection of neurons based on local inhibition - equivalent to multiple independent fully connected weights per neuron (SDBANN)
useLinearSublayers = False

if(useTabularDataset):
	datasetType = "useTabularDataset"
elif(useImageDataset):
	datasetType = "useImageDataset"

#data storage parameters:
workingDrive = '/large/source/ANNpython/FBSAANNpt/'
dataDrive = workingDrive	#'/datasets/'
modelName = 'modelFBSAANN'

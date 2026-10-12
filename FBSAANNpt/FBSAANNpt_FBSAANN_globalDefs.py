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

fbsaOptimiseTrainingIterations = False	#default: True	#orig: False
if(fbsaOptimiseTrainingIterations):
	#LREANN convergence policy: replaces fixed epochs and the epoch LR scheduler.
	#True selects/stops on full training loss; False uses a separate validation set.
	trainSetLossOptimisation = False	#default: False
	fbsaValidationSplitSize = 0.2	#fraction of training rows held out before preprocessing/repetition
	fbsaValidationSplitSeed = 20260930
	fbsaEvaluateEveryIterations = 100
	fbsaMinimumTrainingIterations = 2000	#also require at least one complete training pass
	fbsaTrainingPatience = 2000	#iterations without significant loss improvement
	fbsaTrainingMinDelta = 0.0001
	fbsaTrainingRelativeMinDelta = 0.001
	fbsaTrainingLossGoal = 0.01	#training-loss mode only; also requires 100% train accuracy
	fbsaValidationPatience = 2000
	fbsaValidationMinDelta = 0.0001
	fbsaValidationRelativeMinDelta = 0.001
	fbsaTrainingLearningRateFactor = 0.2
	fbsaTrainingLearningRateReductions = 3
	fbsaTrainingMaxNumericalRecoveries = 5

#Self-Assembling Learning (SAL/SAM), PNAS 2026, symmetric-V/Vbar update.
#Reference: liaoq/SAM2026/run_netv_compare_with_bp.py, REPRO defaults and
#update_layer_gradients_combined_norm_nl (equivalent to "reference" for these settings).
fbsaImplementation = "SAM2026-symmetric-vv"
fbsaReferenceCommit = "8597a1c68a832a12a28997aebecc5c369cf7c15b"
fbsaBackwardWidthMultiplier = 1.0	#hidden feedback width / forward width; input/output widths are unchanged
#Mapping to the reference: lr_w, lr_w_prime, lr_v, lr_v_prime.
fbsaLearningRateFactors = {"forward": 1.0, "backward": 0.2, "lateralForward": 1.0, "lateralBackward": 4.0}
fbsaMomentum = 0.9
fbsaWeightDecay = 0.001	#W and Wprime; decay is applied AFTER scaling each local gradient
fbsaWeightDecayV = None	#None inherits fbsaWeightDecay, as in the reference
fbsaWeightDecayVBar = None	#None inherits the resolved fbsaWeightDecayV
#The base learningRate, batch size, architecture and scheduler still come from ANNpt_globalDefs.
#No normalization; identity backward activation. The forward update DOES include
#the hidden activation derivative; the independent backward pathway does not.
#See FBSAANNpt_PNAS2026.txt for reference mapping and wrapper/template adaptations.
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

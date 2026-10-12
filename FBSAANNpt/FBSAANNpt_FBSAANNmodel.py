"""FBSAANNpt_FBSAANNmodel.py

# Author:
Richard Bruce Baxter - Copyright (c) 2026 BAI Research (bairesearch.com.au)

# License:
MIT License

# Installation:
see ANNpt_main.py

# Usage:
see ANNpt_main.py

# Description:
FBSAANNpt Forward Backward Self-Assembly artificial neural network (FBSAANN) model

"""

import torch as pt
import math
from torch import nn
from ANNpt_globalDefs import *
from torchmetrics.classification import Accuracy
import ANNpt_linearSublayers

class FBSAANNconfig():
	def __init__(self, batchSize, numberOfLayers, numberOfConvlayers, hiddenLayerSize, CNNhiddenLayerSize, inputLayerSize, outputLayerSize, linearSublayersNumber, numberOfFeatures, numberOfClasses, datasetSize, numberOfClassSamples, useFBSA=useFBSA, backwardWidthMultiplier=fbsaBackwardWidthMultiplier):
		self.batchSize = batchSize
		self.numberOfLayers = numberOfLayers
		self.numberOfConvlayers = numberOfConvlayers
		self.hiddenLayerSize = hiddenLayerSize
		self.CNNhiddenLayerSize = CNNhiddenLayerSize
		self.inputLayerSize = inputLayerSize
		self.outputLayerSize = outputLayerSize
		self.linearSublayersNumber = linearSublayersNumber
		self.numberOfFeatures = numberOfFeatures
		self.numberOfClasses = numberOfClasses
		self.datasetSize = datasetSize		
		self.numberOfClassSamples = numberOfClassSamples
		self.useFBSA = useFBSA
		self.backwardWidthMultiplier = backwardWidthMultiplier
		self.fbsaLearningRateFactors = dict(fbsaLearningRateFactors)
		self.fbsaWeightDecay = fbsaWeightDecay

class BiasLayer(nn.Module):
	def __init__(self, num_features):
		super(BiasLayer, self).__init__()
		self.bias = nn.Parameter(pt.zeros(num_features), requires_grad=True)

	def forward(self, x):
		# Adding bias to every element of the input tensor x
		return x + self.bias.unsqueeze(0)
			
class FBSAANNmodel(nn.Module):
		
	def __init__(self, config):
		super().__init__()
		self.config = config
		self.useFBSA = config.useFBSA
		if(config.numberOfLayers < 1):
			raise ValueError("numberOfLayers must include at least the output layer")
		if(self.useFBSA):
			self.validateFBSAconfig()
			
		layersLinearListF = []	#weights forward
		layersLinearListF.append(None)	#input layer i=0
		layersListB = []	#biases
		layersListB.append(None)	#input layer i=0	

		self.n_h = self.generateLayerSizeList()
		print("self.n_h = ", self.n_h)
		
		self.batchNorm = None
		self.batchNormFC = None
		self.dropout = None
		self.maxPool = None

		if(useCNNlayers):
			if CNNbatchNorm:
				self.batchNorm = nn.ModuleList()
				for layerIndex in range(config.numberOfConvlayers):
					_, in_channels, out_channels, _, _, _, _, _ = ANNpt_linearSublayers.getCNNproperties(self, layerIndex)
					self.batchNorm.append(nn.BatchNorm2d(out_channels))
			if batchNormFC:
				self.batchNormFC = nn.ModuleList()
				for layerIndex in range(config.numberOfConvlayers, config.numberOfLayers):
					# determine number of features out of that FC
					if layerIndex == config.numberOfLayers-1:
						feat = config.outputLayerSize
					else:
						feat = config.hiddenLayerSize
					self.batchNormFC.append(nn.BatchNorm1d(feat))
			if(dropout):
				self.dropout = nn.ModuleList()
				for _ in range(config.numberOfLayers):
					self.dropout.append(nn.Dropout2d(p=dropoutProb))
			if(CNNmaxPool):
				self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

		for l2 in range(1, config.numberOfLayers+1):	
			if(supportSkipLayers):
				layersLinearListF2 = []
				for l1 in range(0, l2):
					if(useCNNlayers and l2<=config.numberOfConvlayers and l1==l2-1 and l2<config.numberOfLayers):	#currently only use CNN for non-skip connections	#do not use CNN for final layer
						linearF2 = ANNpt_linearSublayers.generateLinearLayerCNN(self, l1, config, forward=True, bias=False, layerIndex2=l2)	#need to add bias after skip layer connections have been added
					else:
						linearF2 = ANNpt_linearSublayers.generateLinearLayer(self, l1, config, forward=True, bias=False, layerIndex2=l2)	#need to add bias after skip layer connections have been added
					layersLinearListF2.append(linearF2)
				layersLinearF2 = nn.ModuleList(layersLinearListF2)
				layersLinearListF.append(layersLinearF2)
				B = BiasLayer(self.n_h[l2])	#need to add bias after skip layer connections have been added
				layersListB.append(B)
			else:
				l1 = self.getLayerIndex(l2)
				if(useCNNlayers and l2<=config.numberOfConvlayers and l2<config.numberOfLayers):	#do not use CNN for final layer
					linearF = ANNpt_linearSublayers.generateLinearLayerCNN(self, l1, config, forward=True, bias=True, layerIndex2=l2)
				else:
					linearF = ANNpt_linearSublayers.generateLinearLayer(self, l1, config, forward=True, bias=True, layerIndex2=l2)
				layersLinearListF.append(linearF)
		self.layersLinearF = nn.ModuleList(layersLinearListF)
		self.layersB = nn.ModuleList(layersListB)
		self.activationF = ANNpt_linearSublayers.generateActivationFunction(activationFunctionTypeForward)
		#Build the auxiliary circuit after all forward weights, so equal seeds give
		#identical forward initialisation for SAL and the backprop baseline.
		self.layersLinearB = nn.ModuleList()
		self.layersLateralF = nn.ModuleList()
		self.layersLateralB = nn.ModuleList()
		if(self.useFBSA):
			self.generateFBSAnetworks()

		if(useInbuiltCrossEntropyLossFunction):
			self.lossFunctionFinal = nn.CrossEntropyLoss()
		else:
			self.lossFunctionFinal = nn.NLLLoss()	#nn.CrossEntropyLoss == NLLLoss(log(softmax(x)))
		self.accuracyFunction = Accuracy(task="multiclass", num_classes=self.config.outputLayerSize, top_k=1)
		
		self.Ztrace = [None]*(config.numberOfLayers+1)
		self.Atrace = [None]*(config.numberOfLayers+1)
		self.AtraceBackward = [None]*(config.numberOfLayers+1)

	def validateFBSAconfig(self):
		"""The paper specifies dense adjacent-layer motifs, not convolutional updates."""
		if(useCNNlayers or supportSkipLayers or supportSkipLayersResidual or useLinearSublayers or simulatedDendriticBranches):
			raise ValueError("useFBSA=True supports dense adjacent layers only; disable CNN, skip/residual connections and linear sublayers")
		if(activationFunctionTypeBackward != "none"):
			raise ValueError("FBSA implements the paper's identity downstream activation (activationFunctionTypeBackward='none')")
		if(not useInbuiltCrossEntropyLossFunction):
			raise ValueError("FBSA requires cross entropy on output logits")
		if(not math.isfinite(self.config.backwardWidthMultiplier) or self.config.backwardWidthMultiplier <= 0):
			raise ValueError("backwardWidthMultiplier must be finite and positive")
		if(not math.isfinite(self.config.fbsaWeightDecay) or self.config.fbsaWeightDecay < 0):
			raise ValueError("fbsaWeightDecay must be finite and nonnegative")
		for name in ("forward", "backward", "lateralForward", "lateralBackward"):
			factor = self.config.fbsaLearningRateFactors[name]
			if(not math.isfinite(factor) or factor < 0):
				raise ValueError("FBSA learning-rate factors must be finite and nonnegative")

	def generateFBSAnetworks(self):
		"""Register independently initialised feedback and lateral connections.

		For the edge l1 -> l2, layersLinearF[l2] is W^l1 and
		layersLinearB[l2] is Wbar^l1. layersLateralF[l1] is V^l1
		(forward -> backward); layersLateralB[l2] is Vbar^l1
		(backward -> forward). This destination indexing resolves the shifted
		Vbar indices in the paper. The output Vbar is a fixed identity boundary,
		as used in Appendix A; no lateral map is needed below the input.
		"""
		depth = self.config.numberOfLayers
		self.n_hBackward = [self.n_h[0]] + [max(1, math.ceil(width*self.config.backwardWidthMultiplier)) for width in self.n_h[1:-1]] + [self.n_h[-1]]
		self.layersLinearB.append(None)
		self.layersLateralB.append(None)
		for l1 in range(depth):
			l2 = l1+1
			self.layersLinearB.append(nn.Linear(self.n_hBackward[l2], self.n_hBackward[l1], bias=False))
			self.layersLateralF.append(nn.Linear(self.n_h[l1], self.n_hBackward[l1], bias=False))
			if(l2 == depth):
				self.layersLateralB.append(nn.Identity())
			else:
				self.layersLateralB.append(nn.Linear(self.n_hBackward[l2], self.n_h[l2], bias=False))
		self.layersLateralF.append(None)

	def generateLayerSizeList(self):
		n_h = [None]*(self.config.numberOfLayers+1)
		for l2 in range(0, self.config.numberOfLayers+1):
			if(l2 == 0):
				n_h[l2] = self.config.inputLayerSize
			elif(l2 == self.config.numberOfLayers):
				n_h[l2] = self.config.outputLayerSize
			else:
				if(useCNNlayers and l2<=self.config.numberOfConvlayers):
					n_h[l2] = self.config.CNNhiddenLayerSize
				else:
					n_h[l2] = self.config.hiddenLayerSize
		return n_h
			
	def forward(self, trainOrTest, x, y, optim, layer=None):
		#Only the baseline builds an autograd graph. FBSA updates are explicit local
		#correlations, and evaluation never invokes the feedback circuit or optimiser.
		with pt.set_grad_enabled(pt.is_grad_enabled() and trainOrTest and not self.useFBSA):
			loss, accuracy = self.forwardNetwork(x, y)
			if(trainOrTest and self.useFBSA):
				if(optim is None):
					raise ValueError("FBSA training requires an optimiser from FBSAANNpt_FBSAANN.createOptimiser")
				self.updateFBSAweights(y, optim)
			return loss, accuracy

	def forwardNetwork(self, x, y):
			
		outputPred = None
		outputTarget = None

		batch_size = x.shape[0]
		if(useImageDataset):
			#model code always assumes data dimensions are flattened;
			x = x.reshape(batch_size, -1)

		for l2 in range(1, self.config.numberOfLayers+1):
			self.Ztrace[l2] = x.new_zeros([batch_size, self.n_h[l2]])
			self.Atrace[l2] = x.new_zeros([batch_size, self.n_h[l2]])

		outputPred = x #in case layer=0

		AprevLayer = x
		self.Atrace[0] = AprevLayer
		self.Ztrace[0] = pt.zeros_like(AprevLayer)	#set to zero as not used (just used for shape initialisation)
		
		maxLayer = self.config.numberOfLayers
		
		for l2 in range(1, maxLayer+1):
			#print("l2 = ", l2)
			#print("self.config.numberOfLayers = ", self.config.numberOfLayers)
			
			A, Z = self.neuralNetworkPropagationLayerForward(l2, AprevLayer)
			
			if(l2 == self.config.numberOfLayers):
				outputPred = Z	#activation function softmax is applied by self.lossFunctionFinal = nn.CrossEntropyLoss()
			else:
				outputPred = A
			if(l2 == self.config.numberOfLayers):
				outputTarget = y
				loss, accuracy = self.calculateLossAccuracy(outputPred, outputTarget, self.lossFunctionFinal, calculateAccuracy=True)
					
			AprevLayer = A
			self.Ztrace[l2] = Z
			self.Atrace[l2] = A
	
		return loss, accuracy

	def getLayerIndex(self, l2):
		l1 = l2-1
		return l1
	
	def calcMSEDerivative(self, pred, target):
		mse_per_neuron = (pred - target)
		return mse_per_neuron

	def calculateLossAccuracy(self, pred, target, lossFunction, calculateAccuracy=False):
		accuracy = 0
		if(calculateAccuracy):
			accuracy = self.accuracyFunction(pred, target)
		loss = lossFunction(pred, target)
		return loss, accuracy

	def neuralNetworkPropagationLayerForward(self, l2, AprevLayer):

		#print("neuralNetworkPropagationLayerForward l2 = ", l2)

		if(supportSkipLayers):
			Z = pt.zeros_like(self.Ztrace[l2])
			for l1 in range(0, l2):
				cnn = False
				if(useCNNlayers and l2<=self.config.numberOfConvlayers and l1==l2-1 and l2<self.config.numberOfLayers):	#currently only use CNN for non-skip connections #do not use CNN for final layer
					cnn = True
				Zpartial = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l2), self.Atrace[l1], self.layersLinearF[l2][l1], cnn=cnn)
				if(l1 == self.getLayerIndex(l2)):	#is cnn layer
					Zpartial = ANNpt_linearSublayers.convReshapeIntermediate(self, self.getLayerIndex(l2), Zpartial)
					Zpartial = ANNpt_linearSublayers.executeMaxPoolLayer(self, self.getLayerIndex(l2), Zpartial, self.maxPool)	#maxPool
					Zpartial = ANNpt_linearSublayers.convReshapeFinal(self, self.getLayerIndex(l2), Zpartial)
				Z = pt.add(Z, Zpartial)
			Z = self.layersB[l2](Z)	#add the bias once after summing skip connections
			maxPoolAfterActivation = False	#maxPool operation already applied
		else:
			cnn = False
			if(useCNNlayers and l2<=self.config.numberOfConvlayers and l2<self.config.numberOfLayers):	#do not use CNN for final layer
				cnn = True
			Z = ANNpt_linearSublayers.executeLinearLayer(self, self.getLayerIndex(l2), AprevLayer, self.layersLinearF[l2], cnn=cnn)
			maxPoolAfterActivation = True

		Z = ANNpt_linearSublayers.executeResidual(self, self.getLayerIndex(l2), l2, Z, AprevLayer)	#residual
		Z = ANNpt_linearSublayers.convReshapeIntermediate(self, self.getLayerIndex(l2), Z, supportMaxPool=maxPoolAfterActivation)
		Z = ANNpt_linearSublayers.executeBatchNormLayer(self, self.getLayerIndex(l2), Z, self.batchNorm, self.batchNormFC)	#batchNorm
		A = ANNpt_linearSublayers.executeActivationLayer(self, self.getLayerIndex(l2), Z, self.activationF)	#relU
		A = ANNpt_linearSublayers.executeDropoutLayer(self, self.getLayerIndex(l2), A, self.dropout)	#dropout
		if(maxPoolAfterActivation):
			A = ANNpt_linearSublayers.executeMaxPoolLayer(self, self.getLayerIndex(l2), A, self.maxPool)	#maxPool
		A = ANNpt_linearSublayers.convReshapeFinal(self, self.getLayerIndex(l2), A)

		return A, Z

	def neuralNetworkPropagationLayerBackward(self, l2, AprevLayer):
		#Equation (2), D_d = I. In particular, this is neither W.T nor ReLU(error).
		return self.layersLinearB[l2](AprevLayer)

	@pt.no_grad()
	def updateFBSAweights(self, target, optim):
		"""Minibatch equations (5)-(8), https://arxiv.org/abs/2412.20018.

		Activities are row vectors here. The output error is the negative CE
		derivative per example; average outer products exactly once over the batch.
		Equation (8) uses post-activation h (the appendix pseudocode omits D_u).
		Biases in the template's forward network act as constant-input synapses.
		All directions use OLD weights/activities, then one SGD step applies them
		simultaneously, including uniform decay. No backward()/autograd.grad() call.
		"""
		optim.zero_grad(set_to_none=True)
		depth = self.config.numberOfLayers
		logits = self.Ztrace[depth]
		batch_size = logits.shape[0]
		error = -pt.softmax(logits, dim=1)
		error.scatter_add_(1, target.unsqueeze(1), pt.ones_like(target, dtype=logits.dtype).unsqueeze(1))
		self.AtraceBackward[depth] = error
		for l2 in range(depth, 0, -1):
			self.AtraceBackward[l2-1] = self.neuralNetworkPropagationLayerBackward(l2, self.AtraceBackward[l2])

		factors = self.config.fbsaLearningRateFactors
		for l2 in range(1, depth+1):
			l1 = l2-1
			h1 = self.Atrace[l1]
			h2 = logits if l2 == depth else self.Atrace[l2]
			b1 = self.AtraceBackward[l1]
			b2 = self.AtraceBackward[l2]
			projectedError = self.layersLateralB[l2](b2)
			projectedActivity = self.layersLateralF[l1](h1)
			forwardLayer = self.layersLinearF[l2]
			forwardLayer.weight.grad = -(factors["forward"]/batch_size) * (projectedError.T @ h1)
			if(forwardLayer.bias is not None and forwardLayer.bias.requires_grad):
				forwardLayer.bias.grad = -factors["forward"] * projectedError.mean(dim=0)
			self.layersLinearB[l2].weight.grad = -(factors["backward"]/batch_size) * (projectedActivity.T @ b2)
			self.layersLateralF[l1].weight.grad = -(factors["lateralForward"]/batch_size) * (b1.T @ h1)
			if(l2 < depth):
				self.layersLateralB[l2].weight.grad = -(factors["lateralBackward"]/batch_size) * (h2.T @ b2)
		optim.step()


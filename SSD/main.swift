// Copyright 2020 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import TensorFlow
import Batcher

/// A labeled example (or batch thereof) for object detection with SSD.
///
/// For now, we leave most of the work to existing Python code and run the Swift model off a dataset
/// that has already undergone dataset augmentation and had its target boxes matched to anchors.
public struct LabeledSsdExample {
    /// The transformed image's pixel data, shape [batchSize, height, width, 3].
    // TODO: To which numeric range are the pixel values normalized?
    public var image: Tensor<Float>

    /// The target class for each anchor box, shape [batchSize, numAnchors, 1].
    ///
    /// Most boxes are unused and receive label -1.
    /// For the others, the value is in range 0..<numClasses.
    //
    // TODO: In which order are the anchor boxes listed?
    public var clsLabels: Tensor<Int32>

    /// The target box for each anchor box, shape [batchSize, numAnchors, 4].
    ///
    /// Boxes with target class -1 are to be ignored.
    //
    // TODO: How is this encoded?
    public var boxLabels: Tensor<Float>
}

/// For use with Batcher.
extension LabeledSsdExample: Collatable {
    public init(collating: [Self]) {
	// Collate component-wise.
	self.image = Tensor<Float>(collating: collating.map { x in x.image } )
	self.clsLabels = Tensor<Int32>(collating: collating.map { x in x.clsLabels } )
	self.boxLabels = Tensor<Float>(collating: collating.map { x in x.boxLabels } )
    }
}

/// A dataset for object detection with SSD.
public protocol ObjectDetectionBatchers {
    associatedtype C: Collection  where C.Index == Int  // Batcher's requirement.
    init()
    var training: Batcher<C> { get }
    var test: Batcher<C> { get }
}

let batchSize = 8

/// A dummy dataset that is merely good enough to compile.
struct DummyBatchers: ObjectDetectionBatchers {
    /// The training part of this dataset.
    public let training: Batcher<[LabeledSsdExample]>

    /// The test part of this dataset.
    public let test: Batcher<[LabeledSsdExample]>

    init() {
	var basicDataset = [LabeledSsdExample]()
	for _ in 0..<3 {
	    basicDataset += [
		LabeledSsdExample(
                    image: Tensor<Float>(repeating: -0.123, shape: [300, 300, 3]),
		    clsLabels: Tensor<Int32>(repeating: 2, shape: [8732]),
		    boxLabels: Tensor<Float>(repeating: 0.1, shape: [8732, 4]))
	    ]
	}
        training = Batcher(on: basicDataset, batchSize: batchSize, numWorkers: 1, shuffle: true)
        test = Batcher(on: basicDataset, batchSize: batchSize, numWorkers: 1)
    }
}

let batchers = DummyBatchers()  // We'd rather have COCO.

/// The output of SSDModel.
//
// We cannot conform an output tuple to Differentiable, so it's a struct (last checked Jan 2020).
public struct SSDModelOutput: Differentiable {
    /// The logits for each box, shape [batchSize, numAnchors, numClasses].
    //
    // TODO: How are the boxes ordered?
    var clsOutputs: Tensor<Float>

    /// The changes to each anchor box, shape [batchSize, numAnchors, 4].
    //
    // TODO: How are the boxes ordered?
    var boxOutputs: Tensor<Float>
}

public struct PadConvBnRelu: Layer {  // conv_fixed_padding + batch_norm_relu
    public var pad = [ZeroPadding2D<Float>]() // TODO(TF-499): Use ZeroPadding2D<Float>? when ready.
    public var conv: Conv2D<Float>
    public var bn: BatchNorm<Float>
    @noDerivative
    public var useRelu: Bool

    public init(
        filterShape: (Int, Int, Int, Int),
        stride: Int = 1,
	useRelu: Bool = true,
	bnInitZero: Bool = false
    ) {
	var padding: Padding
	if stride > 1 {
	    let kernelSize = filterShape.0
	    assert(kernelSize == filterShape.1, "filterShape.1 != .0")
	    let padTotal = kernelSize - 1
	    let padBegin = padTotal / 2
	    let padEnd = padTotal - padBegin
	    pad += [ZeroPadding2D<Float>(padding: ((padBegin, padEnd), (padBegin, padEnd)))]
	    padding = .valid
	} else {
	    padding = .same
	}
        self.conv = Conv2D(filterShape: filterShape, strides: (stride, stride), padding: padding)
        self.bn = BatchNorm<Float>(featureCount: filterShape.3)
	if bnInitZero { self.bn.scale = Tensor<Float>(zeros: [filterShape.3]) }
	self.useRelu = useRelu
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
	var tmp = input
	if pad.count == 1 {
	    tmp = pad[0](tmp)
	}
        tmp = tmp.sequenced(through: conv, bn)
	if useRelu { tmp = relu(tmp) }
	return tmp
    }
}



struct ResNetBlock: Layer {   // residual_block aka ResidualBasicBlock[Shortcut]
    var projection = [PadConvBnRelu]()  // TODO(TF-499): Use PadConvBnRelu? when ready.
    var layer1: PadConvBnRelu
    var layer2: PadConvBnRelu

    public init(featureCount: Int, filters: Int, stride: Int, useProjection: Bool = false) {
	if useProjection {
	    projection += [PadConvBnRelu(
		    filterShape: (1, 1, featureCount, filters), stride: stride, useRelu: false)
	    ]
	}
	layer1 = PadConvBnRelu(
	    filterShape: (3, 3, featureCount, filters), stride: stride)
	layer2 = PadConvBnRelu(
	    filterShape: (3, 3, filters, filters), stride: 1, useRelu: false, bnInitZero: true)
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
	var shortcut = input
	if projection.count == 1 {
	    shortcut = projection[0](input)
	}
	let tmp = input.sequenced(through: layer1, layer2)
	return relu(tmp + shortcut)
    }
}


struct ResNetBlockGroup: Layer {  // aka ResidualBasicBlock[Shortcut] + ResidualBasicBlockStack
    var blocks: [ResNetBlock]

    public init(
	featureCount: Int, filters: Int, blocksCount: Int, stride: Int, useProjection: Bool = true
    ) {
	let block0 = ResNetBlock(
	    featureCount: featureCount, filters: filters,
	    stride: stride, useProjection: useProjection)
	blocks = [block0]
	for _ in 1..<blocksCount {
	    blocks += [ResNetBlock(featureCount: filters, filters: filters, stride: 1)]
	}
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let blocksReduced = blocks.differentiableReduce(input) { last, layer in
            layer(last)
        }
        return blocksReduced
    }
}

let numDefaults = [4, 6, 6, 6, 4, 4]  // Indexed by zero-based level.
let featuresHiddenDim = [-1, 256, 256, 128, 128, 128]
let featuresOutputDim = [256, 512, 512, 256, 256, 256]

/// The SSDModel for object detection (incomplete).
struct SSDModel: Layer {
    /// The number of object classes in the dataset.
    @noDerivative
    public var numClasses: Int

    var pad0 = ZeroPadding2D<Float>(padding: (3, 3))  // For kernelSize 7.
    // TODO: Remove bias term in conv0.
    // TODO: Change weight initialization from default glorotUniform to truncated normal.
    var conv0 = Conv2D<Float>(filterShape: (7, 7, 3, 64), strides: (2, 2))
    var bn0 = BatchNorm<Float>(featureCount: 64, momentum: 0.9, epsilon: 1e-5)
    var relu0 = Function<Tensor<Float>, Tensor<Float>>(relu)
    var pool0 = MaxPool2D<Float>(poolSize: (3, 3), strides: (2, 2), padding: .same)

    var block2 = ResNetBlockGroup(
	featureCount:  64, filters:  64, blocksCount: 3, stride: 1, useProjection: false)
    var block3 = ResNetBlockGroup(
	featureCount:  64, filters: 128, blocksCount: 4, stride: 2)
    var block4 = ResNetBlockGroup(
	featureCount: 128, filters: featuresOutputDim[0], blocksCount: 6, stride: 1)

   
    // feats4
    var block7conv1x1 = Conv2D<Float>(
	filterShape: (1, 1, 256, featuresHiddenDim[1]), padding: .same, activation: relu)
    var block7conv3x3 = Conv2D<Float>(
	filterShape: (3, 3, featuresHiddenDim[1], featuresOutputDim[1]),
	strides: (2, 2), padding: .same, activation: relu)

    // feats5
    var block8conv1x1 = Conv2D<Float>(
	filterShape: (1, 1, featuresOutputDim[1], featuresHiddenDim[2]),
	padding: .same, activation: relu)
    var block8conv3x3 = Conv2D<Float>(
	filterShape: (3, 3, featuresHiddenDim[2], featuresOutputDim[2]),
	strides: (2, 2), padding: .same, activation: relu)

    // feats6
    var block9conv1x1 = Conv2D<Float>(
	filterShape: (1, 1, featuresOutputDim[2], featuresHiddenDim[3]),
	padding: .same, activation: relu)
    var block9conv3x3 = Conv2D<Float>(
	filterShape: (3, 3, featuresHiddenDim[3], featuresOutputDim[3]),
	strides: (2, 2), padding: .same, activation: relu)

    // feats7
    var block10conv1x1 = Conv2D<Float>(
	filterShape: (1, 1, featuresOutputDim[3], featuresHiddenDim[4]),
	padding: .same, activation: relu)
    var block10conv3x3 = Conv2D<Float>(
	filterShape: (3, 3, featuresHiddenDim[4], featuresOutputDim[4]),
	padding: .valid, activation: relu)

    // feats8
    var block11conv1x1 = Conv2D<Float>(
	filterShape: (1, 1, featuresOutputDim[4], featuresHiddenDim[5]),
	padding: .same, activation: relu)
    var block11conv3x3 = Conv2D<Float>(
	filterShape: (3, 3, featuresHiddenDim[5], featuresOutputDim[5]),
	padding: .valid, activation: relu)


    var classNets = [Conv2D<Float>]()
    var boxNets = [Conv2D<Float>]()
    
    /// Creates a model.
    ///
    /// - Parameter numClasses: The number of object classes in the dataset.
    public init(numClasses: Int) {
        self.numClasses = numClasses
	assert(
	    featuresOutputDim.count == numDefaults.count,
	    "Won't zip counts \(featuresOutputDim.count) vs \(numDefaults.count)")
	for (ftDim, numDefaultBoxes) in zip(featuresOutputDim, numDefaults) {
	    classNets.append(Conv2D<Float>(
		    filterShape: (3, 3, ftDim, numClasses * numDefaultBoxes), padding: .same))
	    boxNets.append(Conv2D<Float>(
		    filterShape: (3, 3, ftDim, 4 * numDefaultBoxes), padding: .same))
	}
    }

    /// Returns predicted boxes and their classes for a batch of images.
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> SSDModelOutput {
	// Index range 0...5 (not 3...8).

	let tmp0 = input.sequenced(through: pad0, conv0, bn0, relu0, pool0)
	let feats0 = tmp0.sequenced(through: block2, block3, block4)
	let feats1 = feats0.sequenced(through: block7conv1x1, block7conv3x3)
	let feats2 = feats1.sequenced(through: block8conv1x1, block8conv3x3)
	let feats3 = feats2.sequenced(through: block9conv1x1, block9conv3x3)
	let feats4 = feats3.sequenced(through: block10conv1x1, block10conv3x3)
	let feats5 = feats4.sequenced(through: block11conv1x1, block11conv3x3)

	// AutoDiff has several issues with arrays (as of Jan 2020), so we resort to
	// straight-line code. Ideally, if zip() had .differentiableMap, one could write
	// let feats = [feats0, feats1, feats2, feats3, feats4, feats5]
	// let clsOutputs = zip(classNets, feats).differentiableMap { net, feat in return net(feat) }
	// let boxOutputs = zip(boxNets, feats).differentiableMap { net, feat in return net(feat) }
	let clsOutputs0 = classNets[0](feats0)
	let clsOutputs1 = classNets[1](feats1)
	let clsOutputs2 = classNets[2](feats2)
	let clsOutputs3 = classNets[3](feats3)
	let clsOutputs4 = classNets[4](feats4)
	let clsOutputs5 = classNets[5](feats5)
	let clsOutputs = (
	    clsOutputs0.reshaped(to: [clsOutputs0.shape[0], -1, numClasses]).concatenated(
		with: clsOutputs1.reshaped(to: [clsOutputs1.shape[0], -1, numClasses]),
		alongAxis: 1)
	    .concatenated(
		with: clsOutputs2.reshaped(to: [clsOutputs2.shape[0], -1, numClasses]),
		alongAxis: 1)
	    .concatenated(
		with: clsOutputs3.reshaped(to: [clsOutputs3.shape[0], -1, numClasses]),
		alongAxis: 1)
	    .concatenated(
		with: clsOutputs4.reshaped(to: [clsOutputs4.shape[0], -1, numClasses]),
		alongAxis: 1)
	    .concatenated(
		with: clsOutputs5.reshaped(to: [clsOutputs5.shape[0], -1, numClasses]),
		alongAxis: 1)
	)

	let boxOutputs0 = boxNets[0](feats0)
	let boxOutputs1 = boxNets[1](feats1)
	let boxOutputs2 = boxNets[2](feats2)
	let boxOutputs3 = boxNets[3](feats3)
	let boxOutputs4 = boxNets[4](feats4)
	let boxOutputs5 = boxNets[5](feats5)
	let boxOutputs = (
	    boxOutputs0.reshaped(to: [boxOutputs0.shape[0], -1, 4]).concatenated(
		with: boxOutputs1.reshaped(to: [boxOutputs1.shape[0], -1, 4]),
		alongAxis: 1)
	    .concatenated(
		with: boxOutputs2.reshaped(to: [boxOutputs2.shape[0], -1, 4]),
		alongAxis: 1)
	    .concatenated(
		with: boxOutputs3.reshaped(to: [boxOutputs3.shape[0], -1, 4]),
		alongAxis: 1)
	    .concatenated(
		with: boxOutputs4.reshaped(to: [boxOutputs4.shape[0], -1, 4]),
		alongAxis: 1)
	    .concatenated(
		with: boxOutputs5.reshaped(to: [boxOutputs5.shape[0], -1, 4]),
		alongAxis: 1)
	)
        return SSDModelOutput(clsOutputs: clsOutputs, boxOutputs: boxOutputs)
    }
}

// TODO: Warm-start from a backbone checkpoint.
var model = SSDModel(numClasses: 10)

@differentiable(wrt: (clsOutputs, boxOutputs))
func detectionLoss(
    clsOutputs: Tensor<Float>,
    boxOutputs: Tensor<Float>,
    clsLabels: Tensor<Int32>,
    boxLabels: Tensor<Float>
) -> Tensor<Float> {
    return (
        meanSquaredError(predicted: boxOutputs, expected: boxLabels) +
        softmaxCrossEntropy(
	    logits: clsOutputs.reshaped(to:[-1, clsOutputs.shape[2]]),
	    labels: clsLabels.reshaped(to:[-1]))
    )
}

// TODO: Add learning rate schedule (ramp-up and decay).
let optimizer = SGD(for: model, learningRate: 0.001, momentum: 0.9)

print("Starting training...")

for epoch in 1...10 {
    Context.local.learningPhase = .training
    var lastLoss: Float = 0
    for batch in batchers.training.sequenced() {
        let (image, boxLabels, clsLabels) = (batch.image, batch.boxLabels, batch.clsLabels)
        let (loss, grad) = valueWithGradient(at: model) { model -> Tensor<Float> in
            let modelOutput = model(image)
            return detectionLoss(
                clsOutputs: modelOutput.clsOutputs, boxOutputs: modelOutput.boxOutputs,
                clsLabels: clsLabels, boxLabels: boxLabels)
        }
        lastLoss = loss.scalar!
        optimizer.update(&model, along: grad)
    }
    print("Completed epoch \(epoch), loss = \(lastLoss)")
}

print("Done.")

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

/// A labeled example (or batch thereof) for object detection with SSD.
///
/// For now, we leave most of the work to existing Python code and run the Swift model off a dataset
/// that has already undergone dataset augmentation and had its target boxes matched to anchors.
public struct LabeledSsdExample: TensorGroup {
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

/// A dataset for object detection with SSD.
public protocol ObjectDetectionDataset {
    init()
    var trainingDataset: Dataset<LabeledSsdExample> { get }
    var testDataset: Dataset<LabeledSsdExample> { get }
    var trainingExampleCount: Int { get }
    var testExampleCount: Int { get }
}

let batchSize = 32

/// A dummy dataset that is merely good enough to compile.
struct DummyDataset: ObjectDetectionDataset {
    /// The training part of this dataset.
    public let trainingDataset: Dataset<LabeledSsdExample>

    /// The test part of this dataset.
    public let testDataset: Dataset<LabeledSsdExample>

    /// The number of training examples.
    public let trainingExampleCount = 1

    /// The number of test examples.
    public let testExampleCount = 1

    init() {
        let dummyExamples = LabeledSsdExample(
            image: Tensor<Float>(repeating: -0.123, shape: [3 * batchSize, 224, 224]),
            clsLabels: Tensor<Int32>(repeating: -1, shape: [3 * batchSize, 64 + 16 + 4, 1]),
            boxLabels: Tensor<Float>(repeating: -1, shape: [3 * batchSize, 64 + 16 + 4, 4])
        )
        trainingDataset = Dataset<LabeledSsdExample>(elements: dummyExamples)
        testDataset = Dataset<LabeledSsdExample>(elements: dummyExamples)
    }
}

let dataset = DummyDataset()  // We'd rather have COCO.

/// The output of SSDModel.
//
// We cannot conform an output tuple to Differentiable, so it's a struct (last checked Jan 2020).
public struct SSDModelOutput: Differentiable {
    var clsOutputs: [Tensor<Float>]
    var boxOutputs: [Tensor<Float>]
}

/// The SSDModel for object detection (incomplete).
struct SSDModel: Layer {
    /// The number of object classes in the dataset.
    @noDerivative
    public var numClasses: Int

    var dummyForAutodiffWhileThereIsNothingElse = Tensor<Float>(ones: [144, 144])

    /// Creates a model.
    ///
    /// - Parameter numClasses: The number of object classes in the dataset.
    public init(numClasses: Int) {
        self.numClasses = numClasses
    }

    /// Returns predicted boxes and their classes for a batch of images.
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> SSDModelOutput {
        var classes = [Tensor<Float>]()
        var boxes = [Tensor<Float>]()
        for i in 0..<3 {
            let numAnchors = 64 >> (2 * i)
            classes.append(Tensor<Float>(repeating: 0, shape: [batchSize, numAnchors, numClasses]))
            boxes.append(Tensor<Float>(repeating: 0, shape: [batchSize, numAnchors, 4]))
        }
        return SSDModelOutput(clsOutputs: classes, boxOutputs: boxes)
    }
}

var model = SSDModel(numClasses: 10)

@differentiable
func detectionLoss(
    clsOutputs: [Tensor<Float>],
    boxOutputs: [Tensor<Float>],
    clsLabels: Tensor<Int32>,
    boxLabels: Tensor<Float>
) -> Tensor<Float> {
    return Tensor<Float>(0.0)
}

// TODO: Add learning rate schedule (ramp-up and decay).
let optimizer = SGD(for: model, learningRate: 0.001, momentum: 0.9)

print("Starting training...")

for epoch in 1...10 {
    Context.local.learningPhase = .training
    let trainingData = dataset.trainingDataset.shuffled(
        sampleCount: dataset.trainingExampleCount, randomSeed: Int64(epoch))
    var lastLoss: Float = 0
    for batch in trainingData.batched(batchSize) {
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

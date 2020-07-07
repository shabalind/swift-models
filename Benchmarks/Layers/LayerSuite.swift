// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

import Benchmark
import Datasets
import ImageClassificationModels
import TensorFlow

func makeRandomTensor(
  batchSize: Int,
  dimensions: [Int],
  device: Device
) -> Tensor<Float> {
  var allDimensions = [batchSize]
  allDimensions.append(contentsOf: dimensions)
  let tensor = Tensor<Float>(
    randomNormal: TensorShape(allDimensions), mean: Tensor<Float>(0.5, on: device),
    standardDeviation: Tensor<Float>(0.1, on: device), seed: (0xffeffe, 0xfffe),
    on: device)
  return tensor
}

func makeForwardBenchmark<CustomLayer, SinkType: Sink>(
  layer makeLayer: @escaping () -> CustomLayer,
  inputDimensions: [Int],
  outputDimensions: [Int],
  sink sinkType: SinkType.Type
) -> ((inout BenchmarkState) throws -> Void)
where
  CustomLayer: Layer,
  CustomLayer.Input == Tensor<Float>,
  CustomLayer.Output == Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar == Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    var layer = makeLayer()
    layer.move(to: device)

    let input = makeRandomTensor(
      batchSize: batchSize,
      dimensions: inputDimensions,
      device: device)

    var sink = SinkType() 

    while true {
      do {
        try state.measure {
          sink.consume(tensor: layer(input))
        }
      } catch {
        if settings.backend == .x10 {
          // A synchronous barrier is needed for X10 to ensure all execution completes
          // before tearing down the model.
          LazyTensorBarrier(wait: true)
        }
        throw error
      }
    }

    // Control-flow never gets here, but this removes the warning 
    // about the sink being never used.
    fatalError("unrechable \(sink)")
  }
}

func makeGradientBenchmark<CustomLayer>(
  layer makeLayer: @escaping () -> CustomLayer,
  inputDimensions: [Int],
  outputDimensions: [Int]
) -> ((inout BenchmarkState) throws -> Void)
where
  CustomLayer: Layer,
  CustomLayer.Input == Tensor<Float>,
  CustomLayer.Output == Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar == Float
{
  return { state in
    let settings = state.settings
    let device = settings.device
    let batchSize = settings.batchSize!
    var layer = makeLayer()
    layer.move(to: device)

    let input = makeRandomTensor(
      batchSize: batchSize,
      dimensions: inputDimensions,
      device: device)
    let output = makeRandomTensor(
      batchSize: batchSize,
      dimensions: outputDimensions,
      device: device)

    var sink: CustomLayer.TangentVector = CustomLayer.TangentVector.zero
    sink.move(to: device)

    while true {
      do {
        try state.measure {
          let result = TensorFlow.gradient(at: layer) { layer -> Tensor<Float> in
            let predicted = layer(input)
            return meanAbsoluteError(predicted: predicted, expected: output)
          }
          // Force materialization of the lazy results.
          sink += result
          LazyTensorBarrier()
        }
      } catch {
        if settings.backend == .x10 {
          // A synchronous barrier is needed for X10 to ensure all execution completes
          // before tearing down the model.
          LazyTensorBarrier(wait: true)
        }
        throw error
      }
    }

    // Control-flow never gets here, but this removes the warning 
    // about the sink being never used.
    fatalError("unrechable \(sink)")
  }
}

protocol Sink {
    init()
    mutating func consume(tensor: Tensor<Float>)
}

struct NoSink: Sink {
    init() {}
    mutating func consume(tensor: Tensor<Float>) {}
} 

struct BarrierSink: Sink {
    init() {}
    mutating func consume(tensor: Tensor<Float>) { LazyTensorBarrier() }
} 

struct BarrierNoWaitSink: Sink {
    init() {}
    mutating func consume(tensor: Tensor<Float>) { LazyTensorBarrier(wait: false) }
}

struct ShapeNoBarrierSink: Sink {
    var shape: TensorShape? = nil
    init() {}
    mutating func consume(tensor: Tensor<Float>) {
        shape = tensor.shape
    }
} 

struct ShapeBarrierSink: Sink {
    var shape: TensorShape? = nil
    init() {}
    mutating func consume(tensor: Tensor<Float>) {
        shape = tensor.shape
        LazyTensorBarrier()
    }
} 

struct ShapeBarrierNoWaitSink: Sink {
    var sink: TensorShape? = nil
    init() {}
    mutating func consume(tensor: Tensor<Float>) {
        sink = tensor.shape
        LazyTensorBarrier(wait: false)
    }
} 

struct AdditiveNoBarrierSink: Sink {
    var sink: Tensor<Float>? = nil
    init() {}
    mutating func consume(tensor: Tensor<Float>) {
        if let value = sink {
            sink = value + tensor
        } else {
            sink = tensor
        }
    }
} 

struct AdditiveBarrierSink: Sink {
    var sink: Tensor<Float>? = nil
    init() {}
    mutating func consume(tensor: Tensor<Float>) {
        if let value = sink {
            sink = value + tensor
        } else {
            sink = tensor
        }
        LazyTensorBarrier()
    }
} 

struct AdditiveBarrierNoWaitSink: Sink {
    var sink: Tensor<Float>? = nil
    init() {}
    mutating func consume(tensor: Tensor<Float>) {
        if let value = sink {
            sink = value + tensor
        } else {
            sink = tensor
        }
        LazyTensorBarrier(wait: false)
    }
} 

func makeLayerSuite<CustomLayer>(
  name: String,
  inputDimensions inp: [Int],
  outputDimensions outp: [Int],
  batchSizes: [Int] = [1],
  backends: [Backend.Value] = [.eager, .x10],
  layer: @escaping () -> CustomLayer
) -> BenchmarkSuite
where
  CustomLayer: Layer,
  CustomLayer.Input == Tensor<Float>,
  CustomLayer.Output == Tensor<Float>,
  CustomLayer.TangentVector.VectorSpaceScalar == Float
{
  let inputString = inp.map { String($0) }.joined(separator: "x")
  let outputString = outp.map { String($0) }.joined(separator: "x")

  return BenchmarkSuite(
    name: "\(name)_\(inputString)_\(outputString)",
    settings: WarmupIterations(10)
  ) { suite in
    for batchSize in batchSizes {
      for backend in backends {

        if backend == .x10 { 

        suite.benchmark(
          "forward_b\(batchSize)_\(backend)_none",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
              layer: layer, inputDimensions: inp, outputDimensions: outp,
              sink: NoSink.self)
        )

        suite.benchmark(
          "forward_b\(batchSize)_\(backend)_barrierwait",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
              layer: layer, inputDimensions: inp, outputDimensions: outp,
              sink: BarrierSink.self)
        )

        suite.benchmark(
          "forward_b\(batchSize)_\(backend)_barriernowait",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
              layer: layer, inputDimensions: inp, outputDimensions: outp,
              sink: BarrierNoWaitSink.self)
        )

        suite.benchmark(
          "forward_b\(batchSize)_\(backend)_shapenobarrier",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
              layer: layer, inputDimensions: inp, outputDimensions: outp,
              sink: ShapeNoBarrierSink.self)
        )

        suite.benchmark(
          "forward_b\(batchSize)_\(backend)_shapebarrierwait",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
              layer: layer, inputDimensions: inp, outputDimensions: outp,
              sink: ShapeBarrierSink.self)
        )

        suite.benchmark(
          "forward_b\(batchSize)_\(backend)_shapebarriernowait",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
              layer: layer, inputDimensions: inp, outputDimensions: outp,
              sink: ShapeBarrierNoWaitSink.self)
        )

        suite.benchmark(
          "forward_b\(batchSize)_\(backend)_addbarrierwait",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
              layer: layer, inputDimensions: inp, outputDimensions: outp,
              sink: AdditiveBarrierSink.self)
        )

        suite.benchmark(
          "forward_b\(batchSize)_\(backend)_addbarriernowait",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
              layer: layer, inputDimensions: inp, outputDimensions: outp,
              sink: AdditiveBarrierNoWaitSink.self)
        )

        } else if backend == .eager {

        suite.benchmark(
          "forward_b\(batchSize)_\(backend)",
          settings: Backend(backend), BatchSize(batchSize),
          function: makeForwardBenchmark(
              layer: layer, inputDimensions: inp, outputDimensions: outp,
              sink: NoSink.self)
        )

        }

        // suite.benchmark(
        //   "forward_and_gradient_b\(batchSize)_\(backend)",
        //   settings: Backend(backend), BatchSize(batchSize),
        //   function: makeGradientBenchmark(
        //     layer: layer, inputDimensions: inp, outputDimensions: outp))
      }
    }
  }
}

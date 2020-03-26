import TensorFlow

@differentiable
func detectionLoss(
    boxOutputs: Tensor<Float>,
    boxLabels: Tensor<Float>
) -> Tensor<Float> {
    return meanSquaredError(predicted: boxOutputs, expected: boxLabels) 
}

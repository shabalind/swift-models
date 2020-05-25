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

import Datasets
import TensorFlow

/// Converts LabeledObjectes for one image into class and box targets for each SSD anchor box.
///
/// - Parameters:
///     - boxLabels: The LabeledObjectes for one image.
///
/// - Returns:
///     - clsLabels: A Tensor<Int32> of shape [numAnchors, 1] with the prediction targets for each
///       anchor box. The target is 0 ("background"), or a real class id starting at 1.
///     - boxLabels: A Tensor<Int32> of shape [numAnchors, 4] with the prediction target
///       (ty, tx, th, tw) of the shape of each anchor box. If the anchor box's class target is 0,
///       this is zeroed out. Otherwise, this is the Faster R-CNN box encoding given by the formulas
///           ty = (y - ya) / ha / scaleXY
///           tx = (x - xa) / wa / scaleXY
///           th = log(h / ha) / scaleHW
///           tw = log(w / wa) / scaleHW
///        where x, y, w, h denote the box's center coordinates, width and height
///        respectively. Similarly, xa, ya, wa, ha denote the anchor's center
///        coordinates, width and height. tx, ty, tw and th denote the anchor-encoded
///        center, width and height respectively.
///        The scale factors are hyperparameters to help shape the loss function.
///        For details, see http://arxiv.org/abs/1506.01497 and Python class FasterRcnnBoxCoder.
public func getSsdTargets(inputBoxes: [LabeledObject])
-> (clsLabels: Tensor<Int32>, boxLabels: Tensor<Float>) {
    let anchors = getDefaultBoxes()
    let matches = match(anchors: anchors, targets: inputBoxes)
    var clsLabels = [Tensor<Int32>]()
    var boxLabels = [Tensor<Float>]()
    var matchedCount = 0
    for (i, j) in matches.enumerated() {
	if j >= 0 {
	    let anchor = anchors[i]
	    let matchedInputBox = inputBoxes[j]
	    let clsLabel = Int32(matchedInputBox.classId)
	    assert(clsLabel > 0, "Invalid classId \(clsLabel) (zero is for background).")
	    clsLabels.append(Tensor<Int32>(clsLabel))
	    boxLabels.append(encodeSsdBoxTarget(targetBox: matchedInputBox, anchorBox: anchor))
	    matchedCount += 1
	} else {
	    clsLabels.append(Tensor<Int32>(Int32(0)))
	    boxLabels.append(Tensor<Float>(repeating: 0.0, shape: [4]))
	}
    }
    if inputBoxes.count == 0 {
	print("#### WEIRD: zero inputBoxes")
    } else if matchedCount == 0 {
	print("#### WEIRD: zero matches for \(inputBoxes.count) inputBoxes: \(inputBoxes)")
    }
    return (Tensor<Int32>(stacking: clsLabels), Tensor<Float>(stacking: boxLabels))
}

/// A purely geometric box without labels etc.
struct Box {
    let xMin: Float
    let xMax: Float
    let yMin: Float
    let yMax: Float
    init(xMin x0: Float, xMax x1: Float, 
         yMin y0: Float, yMax y1: Float) {
       self.xMin = x0
       self.xMax = x1
       self.yMin = y0
       self.yMax = y1
    }
}


/// Clips a float value to the unit interval.
func clip01(_ x: Float) -> Float { return max(0, min(1, x)) }


extension Box {
    init(centerX: Float, centerY: Float, width: Float, height: Float) {
	self.init(
	    xMin: centerX - width/2, xMax: centerX + width/2,
	    yMin: centerY - height/2, yMax: centerY + height/2)
    }
}


/// The "magic numbers" of the SSD model. See Python file ssd_constants.py.
struct SsdConstants {
    static let imageSize = 300
    static let steps = [8, 16, 32, 64, 100, 300]
    static let featureSizes = [38, 19, 10, 5, 3, 1]
    static let scales = [21, 45, 99, 153, 207, 261, 315]
    static let aspectRatios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    static let numDefaults = [4, 6, 6, 6, 4, 4]
    static let numSsdBoxes = 8732
}

/// Returns the anchor boxes of the SSD model, in the linear order expected for training labels.
/// See Python class DefaultBoxes.
func getDefaultBoxes() -> [Box] {
    let fk = SsdConstants.steps.map { SsdConstants.imageSize / $0 }
    var defaultBoxes = [Box]()
    for (idx, featureSize) in SsdConstants.featureSizes.enumerated() {
	let sk1 = Float(SsdConstants.scales[idx]) / Float(SsdConstants.imageSize)  // This level.
	let sk2 = Float(SsdConstants.scales[idx+1]) / Float(SsdConstants.imageSize)  // One level up.
	let sk3 = (sk1 * sk2).squareRoot()  // "Half a level" up (geometric mean).
	var allSizes = [(sk1, sk1), (sk3, sk3)]
	
	for alpha in SsdConstants.aspectRatios[idx] {
	    let w = sk1 * Float(alpha).squareRoot()
	    let h = sk1 / Float(alpha).squareRoot()
	    allSizes += [(w, h), (h, w)]
	}
        assert(allSizes.count == SsdConstants.numDefaults[idx])
	
	for (w, h) in allSizes {
	    for i in 0..<featureSize {
		for j in 0..<featureSize {
		    let cx = Float((Double(j) + 0.5) / Double(fk[idx]))
		    let cy = Float((Double(i) + 0.5) / Double(fk[idx]))
		    defaultBoxes.append(Box(
			    centerX: clip01(cx),
			    centerY: clip01(cy),
			    width: clip01(w),
			    height: clip01(h)))

		}
	    }
	}
    }
    assert(defaultBoxes.count == SsdConstants.numSsdBoxes)
    return defaultBoxes
}


/// Read-only access to box coordinates. Lets us treat Box and LabeledObject uniformly.
protocol ConstantBox {
    var xMin: Float { get }
    var xMax: Float { get }
    var yMin: Float { get }
    var yMax: Float { get }
}

extension LabeledObject: ConstantBox {}
extension Box: ConstantBox {}


/// Returns index of best-matching target for each anchor, or negative for none.
///
/// - Parameters:
///   - anchors: The model's anchor boxes.
///   - targets: The (usually much fewer) boxes to predict.
///
/// - Returns:
///   An array of matches, such that targets[matches[i]] is the chosen match for anchors[i],
////  or matches[i] < 0 to indicate the absense of a match.
func match(anchors: [ConstantBox], targets: [ConstantBox]) -> [Int] {
    let threshold = Float(0.5)
    var matches = [Int](repeating: -1, count: anchors.count)
    for i in 0..<anchors.count {
	var maxIOU = Float(-1.0)
	var maxJ = -1
	for j in 0..<targets.count {
	    let iou = intersectionOverUnion(anchors[i], targets[j])
	    if iou > maxIOU {
		maxIOU = iou
		maxJ = j
	    }
	}
	if maxIOU >= threshold {
	    matches[i] = maxJ
	}
    }
    return matches
}

/// Returns the ratio if intersection over union areas for two boxes (zero if empty intersection).
func intersectionOverUnion(_ a: ConstantBox, _ b: ConstantBox) -> Float {
    // Compute the intersection. Return early if area is zero.
    let xMin = max(a.xMin, b.xMin)
    let xMax = min(a.xMax, b.xMax)
    if xMin >= xMax { return 0.0 }
    let yMin = max(a.yMin, b.yMin)
    let yMax = min(a.yMax, b.yMax)
    if yMin >= yMax { return 0.0 }
    let intersection = (xMax - xMin) * (yMax - yMin)
    // Compute the union.
    let union = ((a.xMax - a.xMin) * (a.yMax - a.yMin) + (b.xMax - b.xMin) * (b.yMax - b.yMin)
                 - intersection)
    if union <= 0.0 { return 0.0 }  // Avoid corner cases.
    return intersection / union
}

extension ConstantBox {
    /// Returns the box's center point (x,y), width w and height h as a quadruple of numbers.
    func toXYWH() -> (Float, Float, Float, Float) {
        let x = (xMax + xMin) / 2.0
	let y = (yMax + yMin) / 2.0
        let w = xMax - xMin
        let h = yMax - yMin
	return (x, y, w, h)
    }
}

/// Encodes box as SSD prediction target for the given anchor.
func encodeSsdBoxTarget(targetBox: ConstantBox, anchorBox: ConstantBox) -> Tensor<Float> {
    let (x, y, w, h) = targetBox.toXYWH()
    let (xa, ya, wa, ha) = anchorBox.toXYWH()
    let ty = (y - ya) / ha
    let tx = (x - xa) / wa
    let th = log(h / ha)
    let tw = log(w / wa)

    let scaleXY = Float(0.1)
    let scaleHW = Float(0.2)
    return Tensor<Float>([ty/scaleXY, tx/scaleXY, th/scaleHW, tw/scaleHW])
}

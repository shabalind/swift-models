import Foundation
import Batcher

public struct LazyImage {
    let width: Int
    let height: Int
    let url: URL

    init(width w: Int, height h: Int, url u: URL) {
        self.width = w
        self.height = h
        self.url = u
    }
}

public struct LabeledBoundingBox {
    let xMin: Float
    let xMax: Float
    let yMin: Float
    let yMax: Float
    let className: String
    let classId: Int
    let isCrowd: Int 
    let area: Float

    init(xMin x0: Float, xMax x1: Float, 
         yMin y0: Float, yMax y1: Float,
         className: String, classId: Int,
         isCrowd: Int, area: Float) {
       self.xMin = x0
       self.xMax = x1
       self.yMin = y0
       self.yMax = y1
       self.className = className
       self.classId = classId
       self.isCrowd = isCrowd
       self.area = area
    }
}

public struct ObjectDetectionExample: Collatable, KeyPathIterable {
    let image: LazyImage
    let boundingBoxes: [LabeledBoundingBox]
}

public protocol ObjectDetectionDataset {
    associatedtype SourceDataSet: Collection
    where SourceDataSet.Element == ObjectDetectionExample, SourceDataSet.Index == Int
    init(batchSize: Int)
    var training: Batcher<SourceDataSet> { get }
    var test: Batcher<SourceDataSet> { get }
}

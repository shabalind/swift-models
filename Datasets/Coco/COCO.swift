import Foundation

/// Coco dataset API that loads annotation file and prepares 
/// data structures for data set access.
public struct COCO {
    public typealias Dataset = [String: Any]
    public typealias Info = [String: Any]
    public typealias Annotation = [String: Any]
    public typealias AnnotationId = Int
    public typealias Image = [String: Any]
    public typealias ImageId = Int
    public typealias Category = [String: Any]
    public typealias CategoryId = Int

    public var dataset: Dataset
    public var info: Info
    public var anns: [AnnotationId: Annotation]
    public var cats: [CategoryId: Category]
    public var imgs: [ImageId: Image]
    public var imgToAnns: [ImageId: [Annotation]]
    public var catToImgs: [CategoryId: [ImageId]]

    public init(fromFile fileURL: URL) throws {
        let contents = try String(contentsOfFile: fileURL.path)
        let data = contents.data(using: .utf8)!
        let parsed = try JSONSerialization.jsonObject(with: data)
        self.dataset = parsed as! Dataset
        self.info = [:]
        self.anns = [:]
        self.cats = [:]
        self.imgs = [:]
        self.imgToAnns = [:]
        self.catToImgs = [:]
        self.createIndex()
    }

    mutating func createIndex() {
        if let info = dataset["info"] {
            self.info = info as! Info
        }
        if let annotations = dataset["annotations"] {
            let anns = annotations as! [Annotation]
            for ann in anns {
                let ann_id = ann["id"] as! AnnotationId
                let image_id = ann["image_id"] as! ImageId
                self.imgToAnns[image_id, default: []].append(ann)
                self.anns[ann_id] = ann
            }
        }
        if let images = dataset["images"] {
            let imgs = images as! [Image]
            for img in imgs {
                let img_id = img["id"] as! ImageId
                self.imgs[img_id] = img
            }
        }
        if let categories = dataset["categories"] {
            let cats = categories as! [Category]
            for cat in cats {
                let cat_id = cat["id"] as! CategoryId
                self.cats[cat_id] = cat
            }
        }
        if let annotations = dataset["annotations"] {
            let anns = annotations as! [Annotation]
            for ann in anns {
                let cat_id = ann["category_id"] as! CategoryId
                let image_id = ann["image_id"] as! ImageId
                self.catToImgs[cat_id, default: []].append(image_id)
            }
        }
    }

    /// Get annotation ids that satisfy given filter conditions.
    func getAnnotationIds(
        imageIds: [ImageId] = [],
        categoryIds: [CategoryId] = [],
        areaRange: [[Double]] = [],
        isCrowd: Int? = nil
    ) -> [AnnotationId] {
        let filterByImageId = imageIds.count != 0
        let filterByCategoryId = imageIds.count != 0
        let filterByAreaRange = areaRange.count != 0
        let filterByIsCrowd = isCrowd != nil

        var anns: [Annotation] = []
        if filterByImageId {
            for imageId in imageIds {
                if let imageAnns = self.imgToAnns[imageId] {
                    for imageAnn in imageAnns {
                        anns.append(imageAnn)
                    }
                }
            }
        } else {
            anns = self.dataset["annotations"] as! [Annotation]
        }

        var annIds: [AnnotationId] = []
        for ann in anns {
            if filterByCategoryId {
                let categoryId = ann["category_id"] as! CategoryId
                if !categoryIds.contains(categoryId) {
                    continue
                }
            }
            if filterByAreaRange {
                let area = ann["area"] as! [Double]
                if !areaLessThan(areaRange[0], area) || !areaLessThan(area, areaRange[1]) {
                    continue
                }
            }
            if filterByIsCrowd {
                let annIsCrowd = ann["iscrowd"] as! Int
                if annIsCrowd != isCrowd! {
                    continue
                }
            }
            let id = ann["id"] as! AnnotationId
            annIds.append(id)
        }
        return annIds
    }

    /// A helper function that decides if one area is less than the other.
    private func areaLessThan(_ left: [Double], _ right: [Double]) -> Bool {
        // TODO: 
        return false
    }

    /// Get category ids that satisfy given filter conditions.
    func getCategoryIds(
        categoryNames: [String] = [],
        supercategoryNames: [String] = [],
        categoryIds: [CategoryId] = []
    ) -> [CategoryId] {
        let filterByName = categoryNames.count != 0
        let filterBySupercategory = supercategoryNames.count != 0
        let filterById = categoryIds.count != 0
        var categoryIds: [CategoryId] = []
        let cats = self.dataset["categories"] as! [Category]
        for cat in cats {
            let name = cat["name"] as! String
            let supercategory = cat["supercategory"] as! String
            let id = cat["id"] as! CategoryId
            if filterByName && !categoryNames.contains(name) {
                continue
            }
            if filterBySupercategory && !supercategoryNames.contains(supercategory) {
                continue
            }
            if filterById && !categoryIds.contains(id) {
                continue
            }
            categoryIds.append(id)
        }
        return categoryIds
    }

    /// Get image ids that satisfy given filter conditions.
    func getImageIds(
        imageIds: [ImageId] = [],
        categoryIds: [CategoryId] = []
    ) -> [ImageId] {
        if imageIds.count == 0 && categoryIds.count == 0 {
            return Array(self.imgs.keys)
        } else {
            var ids = Set(imageIds)
            for (i, catId) in categoryIds.enumerated() {
                if i == 0 && ids.count == 0 {
                    ids = Set(self.catToImgs[catId]!)
                } else {
                    ids = ids.intersection(Set(self.catToImgs[catId]!))
                }
            }
            return Array(ids)
        }
    }

    /// Load annotations with specified ids.
    func loadAnnotations(ids: [AnnotationId] = []) -> [Annotation] {
        var anns: [Annotation] = []
        for id in ids {
            anns.append(self.anns[id]!)
        }
        return anns
    }

    /// Load categories with specified ids.
    func loadCategories(ids: [CategoryId] = []) -> [Category] {
        var cats: [Category] = []
        for id in ids {
            cats.append(self.cats[id]!)
        }
        return cats
    }

    /// Load images with specified ids.
    func loadImages(ids: [ImageId] = []) -> [Image] {
        var imgs: [Image] = []
        for id in ids {
            imgs.append(self.imgs[id]!)
        }
        return imgs
    }

    /// Convert segmentation in an annotation to RLE.
    func annotationToRLE(_ ann: Annotation) -> RLE {
        let imgId = ann["image_id"] as! ImageId
        let img = self.imgs[imgId]!
        let h = img["height"] as! Int
        let w = img["weight"] as! Int
        let segm = ann["segmentation"]
        if let polygon = segm as? [Any] {
            let rles = Mask.fromObject(polygon, width: w, height: h)
            return Mask.merge(rles)
        } else {
            fatalError("todo")
        }
    }

    /// Convert segmentation in an anotation to binary mask.
    func annotationToMask(_ ann: Annotation) -> Mask {
        return Mask(fromRLE: annotationToRLE(ann))
    }

    /// Download images from mscoco.org server.
    func downloadImages() {}
}

public struct Mask {
    init(fromRLE rle: RLE) {}

    static func merge(_ rles: [RLE]) -> RLE {
        fatalError("todo")
    }

    static func fromBoundingBoxes(_ bboxes: [[Double]], width w: Int, height h: Int) -> [RLE] {
        var rles: [RLE] = []
        for bbox in bboxes {
            let rle = RLE(fromBoundingBox: bbox, width: w, height: h)
            rles.append(rle)
        }
        return rles
    }

    static func fromPolygons(_ polys: [[Double]], width w: Int, height h: Int) -> [RLE] {
        var rles: [RLE] = []
        for poly in polys {
            let rle = RLE(fromPolygon: poly, width: w, height: h)
            rles.append(rle)
        }
        return rles
    }

    static func fromUncompressedRLEs(_ arr: [[String: Any]], width w: Int, height h: Int) -> [RLE] {
        fatalError("todo")
    }

    static func fromObject(_ obj: Any, width w: Int, height h: Int) -> [RLE] {
        // encode rle from a list of json deserialized objects
        if let arr = obj as? [[Double]] {
            assert(arr.count > 0)
            if arr[0].count == 4 {
                return fromBoundingBoxes(arr, width: w, height: h)
            } else {
                assert(arr[0].count > 4)
                return fromPolygons(arr, width: w, height: h)
            }
        } else if let arr = obj as? [[String: Any]] {
            assert(arr.count > 0)
            assert(arr[0]["size"] != nil)
            assert(arr[0]["counts"] != nil)
            return fromUncompressedRLEs(arr, width: w, height: h)
            // encode rle from a single json deserialized object
        } else if let arr = obj as? [Double] {
            if arr.count == 4 {
                return fromBoundingBoxes([arr], width: w, height: h)
            } else {
                assert(arr.count > 4)
                return fromPolygons([arr], width: w, height: h)
            }
        } else if let dict = obj as? [String: Any] {
            assert(dict["size"] != nil)
            assert(dict["counts"] != nil)
            return fromUncompressedRLEs([dict], width: w, height: h)
        } else {
            fatalError("input type is not supported")
        }
    }
}

public struct RLE {
    var width: Int = 0
    var height: Int = 0
    var m: Int = 0
    var counts: [UInt32] = []

    init(width w: Int, height h: Int, m: Int, counts: [UInt32]) {
        self.width = w
        self.height = h
        self.m = m
        self.counts = counts
    }

    init(fromBoundingBox bb: [Double], width w: Int, height h: Int) {
        let xs = bb[0]
        let ys = bb[1]
        let xe = bb[2]
        let ye = bb[3]
        let xy: [Double] = [xs, ys, xs, ye, xe, ye, xe, ys]
        self.init(fromPolygon: xy, width: w, height: h)
    }

    init(fromPolygon xy: [Double], width w: Int, height h: Int) {
        // upsample and get discrete points densely along the entire boundary
        var k: Int = xy.count / 2
        var j: Int = 0
        var m: Int = 0
        let scale: Double = 5
        var x = [Int](repeating: 0, count: k + 1)
        var y = [Int](repeating: 0, count: k + 1)
        for j in 0..<k { x[j] = Int(scale * xy[j * 2 + 0] + 0.5) }
        x[k] = x[0]
        for j in 0..<k { y[j] = Int(scale * xy[j * 2 + 1] + 0.5) }
        y[k] = y[0]
        for j in 0..<k { m += max(abs(x[j] - x[j + 1]), abs(y[j] - y[j + 1])) + 1 }
        var u = [Int](repeating: 0, count: m)
        var v = [Int](repeating: 0, count: m)
        m = 0
        for j in 0..<k {
            var xs: Int = x[j]
            var xe: Int = x[j + 1]
            var ys: Int = y[j]
            var ye: Int = y[j + 1]
            let dx: Int = abs(xe - xs)
            let dy: Int = abs(ys - ye)
            var t: Int
            let flip: Bool = (dx >= dy && xs > xe) || (dx < dy && ys > ye)
            if flip {
                t = xs
                xs = xe
                xe = t
                t = ys
                ys = ye
                ye = t
            }
            let s: Double = dx >= dy ? Double(ye - ys) / Double(dx) : Double(xe - xs) / Double(dy)
            if dx >= dy {
                for d in 0...dx {
                    t = flip ? dx - d : d
                    u[m] = t + xs
                    v[m] = Int(Double(ys) + s * Double(t) + 0.5)
                    m += 1
                }
            } else {
                for d in 0...dy {
                    t = flip ? dy - d : d
                    v[m] = t + ys
                    u[m] = Int(Double(xs) + s * Double(t) + 0.5)
                    m += 1
                }
            }
        }
        // get points along y-boundary and downsample
        k = m
        m = 0
        var xd: Double
        var yd: Double
        x = [Int](repeating: 0, count: k)
        y = [Int](repeating: 0, count: k)
        for j in 1..<k {
            if u[j] != u[j - 1] {
                xd = Double(u[j] < u[j - 1] ? u[j] : u[j] - 1)
                xd = (xd + 0.5) / scale - 0.5
                if floor(xd) != xd || xd < 0 || xd > Double(w - 1) { continue }
                yd = Double(v[j] < v[j - 1] ? v[j] : v[j - 1])
                yd = (yd + 0.5) / scale - 0.5
                if yd < 0 { yd = 0 } else if yd > Double(h) { yd = Double(h) }
                yd = ceil(yd)
                x[m] = Int(xd)
                y[m] = Int(yd)
                m += 1
            }
        }
        // compute rle encoding given y-boundary points
        k = m
        var a = [UInt32](repeating: 0, count: k + 1)
        for j in 0..<k { a[j] = UInt32(x[j] * Int(h) + y[j]) }
        a[k] = UInt32(h * w)
        k += 1
        a.sort()
        var p: UInt32 = 0
        for j in 0..<k {
            let t: UInt32 = a[j]
            a[j] -= p
            p = t
        }
        var b = [UInt32](repeating: 0, count: k)
        j = 0
        m = 0
        b[m] = a[j]
        m += 1
        j += 1
        while j < k {
            if a[j] > 0 {
                b[m] = a[j]
                m += 1
                j += 1
            } else {
                j += 1
            }
            if j < k {
                b[m - 1] += a[j]
                j += 1
            }
        }
        self.init(width: w, height: h, m: m, counts: b)
    }
}

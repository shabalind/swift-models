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

import Foundation

enum PrintingStyle {
    case plainText
    case json
}

extension BenchmarkResults {
    func print(using style: PrintingStyle) {
        switch style {
        case .plainText:
            printAsPlainText()
        case .json:
            printAsJSON()
        }
    }

    private func printAsPlainText() {
        let (average, standardDeviation) = statistics(for: self.interpretedTimings)
        let configuration = self.configuration
        let settings = configuration.settings

        switch configuration.variety {
        case .inferenceThroughput:
            Swift.print("Benchmark: \(configuration.name)")
            Swift.print("\tVariety: \(configuration.variety.rawValue)")
            Swift.print("\tAfter \(settings.iterations) iterations:")
            Swift.print(
                "\tSamples per second: \(String(format: "%.2f", average)), standard deviation: \(String(format: "%.2f", standardDeviation))"
            )
        case .trainingTime:
            Swift.print("Benchmark: \(configuration.name)")
            Swift.print("\tVariety: \(configuration.variety.rawValue)")
            Swift.print("\tAfter \(settings.iterations) iterations:")
            Swift.print(
                "\tAverage: \(String(format: "%.2f", average)) ms, standard deviation: \(String(format: "%.2f", standardDeviation)) ms"
            )
        }
    }

    private func printAsJSON() {
        printJSON(self)
    }
}

extension BenchmarkConfiguration {
    func print(using style: PrintingStyle) {
        switch style {
        case .plainText:
            printAsPlainText()
        case .json:
            printAsJSON()
        }
    }

    private func printAsPlainText() {
        var result = ""
        result += "--benchmark "
        result += self.name
        result += " "
        switch self.variety {
        case .trainingTime:
            result += "--training "
        case .inferenceThroughput:
            result += "--inference "
        }
        if settings.batches != -1 {
            result += "--batches \(settings.batches) "
        }
        if settings.batchSize != -1 {
            result += "--batchSize \(settings.batchSize) "
        }
        if settings.iterations != -1 {
            result += "--iterations \(settings.iterations) "
        }
        if settings.epochs != -1 {
            result += "--epochs \(settings.epochs) "
        }
        Swift.print(result)
    }

    private func printAsJSON() {
        printJSON(self)
    }
}

/// Print given encodable value to stdout.
func printJSON<T: Encodable>(_ value: T) {
    let data = try! JSONEncoder().encode(value)
    let json = String(data: data, encoding: .utf8)!
    print(json)
}

/// Provides the average and standard deviation of an array of values.
func statistics(for values: [Double]) -> (average: Double, standardDeviation: Double) {
    guard values.count > 0 else { return (average: 0.0, standardDeviation: 0.0) }
    guard values.count > 1 else { return (average: values.first!, standardDeviation: 0.0) }

    let average = (values.reduce(0.0) { $0 + $1 }) / Double(values.count)

    let standardDeviation = sqrt(
        values.reduce(0.0) { $0 + ($1 - average) * ($1 - average) }
            / Double(values.count - 1))

    return (average: average, standardDeviation: standardDeviation)
}

// 시스템 소리(Zoom·브라우저 등 맥에서 나는 소리) → stdout 에 s16le 16k mono.
// 엔진이 ffmpeg 마이크 대신 이 출력을 그대로 읽는다. 엔진이 처음 쓸 때 swiftc 로 컴파일한다.
// 종료 코드 77 = 화면·시스템 오디오 녹음 권한 없음(또는 캡처 시작 실패).
import Foundation
import ScreenCaptureKit
import CoreMedia

let SR = 16000.0
setvbuf(stdout, nil, _IOFBF, 1 << 16)

final class Cap: NSObject, SCStreamOutput, SCStreamDelegate {
    let q = DispatchQueue(label: "syscap")
    let t0 = Date()
    var n = 0                      // 내보낸 샘플 수
    var last = Date()              // 마지막 실제 버퍼 시각

    func put(_ f: UnsafeBufferPointer<Float>) {
        var s = [Int16](repeating: 0, count: f.count)
        for i in 0..<f.count { s[i] = Int16(max(-1, min(1, f[i])) * 32767) }
        s.withUnsafeBytes { _ = fwrite($0.baseAddress, 1, $0.count, stdout) }
        fflush(stdout)
        n += f.count
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sb.isValid, let bb = CMSampleBufferGetDataBuffer(sb) else { return }
        let bytes = CMBlockBufferGetDataLength(bb)   // mono float32 → 데이터 버퍼가 곧 샘플 배열
        guard bytes > 0 else { return }
        var d = Data(count: bytes)
        let st = d.withUnsafeMutableBytes {
            CMBlockBufferCopyDataBytes(bb, atOffset: 0, dataLength: bytes, destination: $0.baseAddress!)
        }
        guard st == kCMBlockBufferNoErr else { return }
        last = Date()
        d.withUnsafeBytes { put($0.bindMemory(to: Float.self)) }
    }

    // 아무 소리도 안 나면 버퍼가 끊긴다 → 무음을 채워 녹음 길이를 실제 시간에 맞춘다
    func padSilence() {
        guard Date().timeIntervalSince(last) > 1.0 else { return }
        let behind = Int(Date().timeIntervalSince(t0) * SR) - n - Int(0.25 * SR)
        guard behind > 0 else { return }
        let z = [Float](repeating: 0, count: behind)
        z.withUnsafeBufferPointer { put($0) }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        FileHandle.standardError.write("stopped: \(error)\n".data(using: .utf8)!)
        exit(1)
    }
}

let cap = Cap()
do {
    let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
    guard let display = content.displays.first else { exit(77) }
    let cfg = SCStreamConfiguration()
    cfg.capturesAudio = true
    cfg.sampleRate = Int(SR)
    cfg.channelCount = 1
    cfg.excludesCurrentProcessAudio = true
    cfg.width = 2; cfg.height = 2                         // 화면은 안 쓴다
    cfg.minimumFrameInterval = CMTime(value: 1, timescale: 1)
    let stream = SCStream(filter: SCContentFilter(display: display, excludingWindows: []),
                          configuration: cfg, delegate: cap)
    try stream.addStreamOutput(cap, type: .audio, sampleHandlerQueue: cap.q)
    try await stream.startCapture()
    let timer = DispatchSource.makeTimerSource(queue: cap.q)
    timer.schedule(deadline: .now() + 0.25, repeating: 0.25)
    timer.setEventHandler { cap.padSilence() }
    timer.resume()
    signal(SIGPIPE, SIG_DFL)                               // 엔진이 파이프를 닫으면 바로 끝난다
    while true { try await Task.sleep(nanoseconds: 3_600_000_000_000) }
} catch {
    FileHandle.standardError.write("syscap: \(error)\n".data(using: .utf8)!)
    exit(77)
}

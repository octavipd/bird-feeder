#include <opencv2/opencv.hpp>
#include <deque>
#include <iostream>
#include <string>
#include <ctime>

int main() {
    // 1. HARDWARE SYNC
    cv::VideoCapture cap(0, cv::CAP_V4L2); // Force V4L2 driver for better Pi support
    if (!cap.isOpened()) return -1;

    // Set resolution to a balanced 720p - high quality but manageable
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    
    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps < 1 || fps > 100) fps = 30.0; // Fallback

    // 2. QUALITY SETTINGS
    // We use a GStreamer pipeline for Hardware Encoding (H.264)
    // This is the "secret sauce" for smooth, high-quality Pi video.
    auto getWriterPipeline = [&](std::string filename, int f) {
        return "appsrc ! videoconvert ! v4l2h264enc extra-controls=\"controls,video_bitrate=5000000;\" ! " 
               "h264parse ! mp4mux ! filesink location=" + filename;
    };

    cv::Ptr<cv::BackgroundSubtractorMOG2> pBackSub = cv::createBackgroundSubtractorMOG2(300, 32, false);
    cv::VideoWriter writer;
    std::deque<cv::Mat> buffer;
    
    cv::Mat frame, fgMask, smallFrame;
    bool isRecording = false;
    int framesSinceMotion = 0;
    int postRollLimit = (int)fps * 5;

    while (true) {
        if (!cap.read(frame)) break;

        // ANALYSIS (Every 4th frame to keep CPU cool)
        static int count = 0;
        bool motionDetected = false;
        if (count++ % 4 == 0) {
            cv::resize(frame, smallFrame, cv::Size(320, 180));
            pBackSub->apply(smallFrame, fgMask);
            motionDetected = (cv::countNonZero(fgMask) > (smallFrame.total() * 0.005));
        }

        if (motionDetected) {
            framesSinceMotion = 0;
            if (!isRecording) {
                std::string fn = "bird_" + std::to_string(time(0)) + ".mp4";
                // Open using the Hardware Pipeline
                writer.open(getWriterPipeline(fn, (int)fps), 0, fps, frame.size(), true);
                isRecording = true;
                while (!buffer.empty()) {
                    writer.write(buffer.front());
                    buffer.pop_front();
                }
            }
        } else {
            framesSinceMotion++;
        }

        if (isRecording) {
            writer.write(frame);
            if (framesSinceMotion > postRollLimit) {
                isRecording = false;
                writer.release();
            }
        } else {
            // Memory efficient buffering: only store if we have to
            buffer.push_back(frame.clone());
            if (buffer.size() > (int)fps * 2) buffer.pop_front();
        }

        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}
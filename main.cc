#include <opencv2/opencv.hpp>
#include <deque>
#include <iostream>
#include <string>
#include <ctime>

std::string getFilename() {
    time_t now = time(0);
    struct tm tstruct = *localtime(&now);
    char buf[80];
    strftime(buf, sizeof(buf), "videos/bird_%Y%m%d_%H%M%S.mp4", &tstruct);
    return std::string(buf);
}

int main() {
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) return -1;

    // 1. DYNAMIC FPS DETECTION
    // Get hardware FPS (usually 30). Fallback to 30 if driver returns 0.
    double hwFPS = cap.get(cv::CAP_PROP_FPS);
    if (hwFPS < 1) hwFPS = 30.0;
    
    int fps = (int)hwFPS;
    int preRollSecs = 2;
    int postRollSecs = 5;
    
    size_t maxBufferFrames = fps * preRollSecs;
    int postRollLimit = fps * postRollSecs;
    
    // 2. RESOURCE OPTIMIZATIONS
    int analysisFrequency = 5; // Only run motion logic every 5th frame
    int frameCounter = 0;
    
    cv::Ptr<cv::BackgroundSubtractorMOG2> pBackSub = cv::createBackgroundSubtractorMOG2(500, 16, false);
    cv::VideoWriter writer;
    std::deque<cv::Mat> buffer;
    
    cv::Mat frame, fgMask, smallFrame;
    bool isRecording = false;
    int framesSinceMotion = 0;
    bool motionDetected = false;

    std::cout << "Monitoring at " << fps << " FPS. Analysis every " << analysisFrequency << " frames." << std::endl;

    while (true) {
        if (!cap.read(frame)) break;
        frameCounter++;

        // --- STEP A: LIGHTWEIGHT MOTION LOGIC ---
        // We only "think" every 5 frames, but we "see" every frame
        if (frameCounter % analysisFrequency == 0) {
            cv::resize(frame, smallFrame, cv::Size(320, 240)); // Low-res analysis
            pBackSub->apply(smallFrame, fgMask);
            cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);
            
            double motionAmount = (double)cv::countNonZero(fgMask) / fgMask.total();
            motionDetected = (motionAmount > 0.005); 
        }

        // --- STEP B: BUFFER & STATE MACHINE ---
        if (motionDetected) {
            framesSinceMotion = 0;
            if (!isRecording) {
                std::string filename = getFilename();
                // Match the VideoWriter FPS to the actual Hardware FPS
                writer.open(filename, cv::VideoWriter::fourcc('a', 'v', 'c', '1'), fps, frame.size());
                isRecording = true;
                
                while (!buffer.empty()) {
                    writer.write(buffer.front());
                    buffer.pop_front();
                }
            }
        } else {
            framesSinceMotion++;
        }

        // --- STEP C: DISK I/O ---
        if (isRecording) {
            writer.write(frame);
            if (framesSinceMotion > postRollLimit) {
                isRecording = false;
                writer.release();
                std::cout << "Saved: " << getFilename() << std::endl;
            }
        } else {
            // Only buffer if we aren't already recording
            buffer.push_back(frame.clone());
            if (buffer.size() > maxBufferFrames) buffer.pop_front();
        }

        // Minimal wait to keep the loop timing consistent
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}
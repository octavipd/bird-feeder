#include <opencv2/opencv.hpp>
#include <deque>
#include <iostream>
#include <string>
#include <ctime>
#include <sys/stat.h>

// Helper to ensure the directory exists
void ensureDirectory(const std::string& dir) {
    struct stat info;
    if (stat(dir.c_str(), &info) != 0) {
        mkdir(dir.c_str(), 0777);
    }
}

std::string getFilename() {
    time_t now = time(0);
    struct tm tstruct = *localtime(&now);
    char buf[80];
    strftime(buf, sizeof(buf), "videos/bird_%Y%m%d_%H%M%S.mp4", &tstruct);
    return std::string(buf);
}

int main() {
    // 1. Setup - Use a smaller resolution for Pi performance
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // PI OPTIMIZATION: Lower resolution significantly reduces CPU load
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    ensureDirectory("videos");

    int fps = 15;               // 15 FPS is a sweet spot for Pi CPU/Thermal
    int preRollSecs = 2;        
    int postRollSecs = 5;       // Longer post-roll captures more "exits"
    double motionThresh = 0.005; // Lowered to 0.5% for smaller birds

    size_t maxBufferFrames = fps * preRollSecs;
    int postRollLimit = fps * postRollSecs;
    
    // MOG2 can be heavy; 'detectShadows = false' saves cycles
    cv::Ptr<cv::BackgroundSubtractorMOG2> pBackSub = cv::createBackgroundSubtractorMOG2(500, 16, false);
    cv::VideoWriter writer;
    std::deque<cv::Mat> buffer;
    
    cv::Mat frame, gray, fgMask;
    bool isRecording = false;
    int framesSinceMotion = 0;

    std::cout << "Headless monitoring started. Logs saved to stdout." << std::endl;

    while (true) {
        if (!cap.read(frame)) break;

        // --- STEP A: Optimized Motion Detection ---
        // Resize for motion analysis only (saves massive CPU)
        cv::Mat smallFrame;
        cv::resize(frame, smallFrame, cv::Size(320, 240));
        
        pBackSub->apply(smallFrame, fgMask);
        cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);

        double motionAmount = (double)cv::countNonZero(fgMask) / fgMask.total();
        bool motionDetected = (motionAmount > motionThresh);

        // --- STEP B: State Machine & Buffer ---
        if (!isRecording) {
            // Use .clone() sparingly on Pi to avoid memory fragmentation
            buffer.push_back(frame.clone()); 
            if (buffer.size() > maxBufferFrames) buffer.pop_front();
        }

        if (motionDetected) {
            framesSinceMotion = 0;
            if (!isRecording) {
                std::string filename = getFilename();
                std::cout << "[EVENT] Motion! Saving to: " << filename << std::endl;
                
                // PI CODEC: 'H264' or 'X264' is preferred if available via FFMPEG
                // Otherwise 'avc1' or 'mp4v' are safe fallbacks.
                writer.open(filename, cv::VideoWriter::fourcc('a','v','c','1'), fps, frame.size());
                
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
                std::cout << "[INFO] Sequence finished." << std::endl;
                isRecording = false;
                writer.release();
            }
        }

        // NO cv::imshow() for headless. 
        // We use a small sleep to prevent 100% CPU usage if the camera is too fast
        if (cv::waitKey(1) == 27) break; 
    }

    return 0;
}
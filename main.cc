#include <opencv2/opencv.hpp>
#include <deque>
#include <iostream>
#include <string>
#include <ctime>

// Helper to generate a timestamped filename
std::string getFilename() {
    time_t now = time(0);
    struct tm tstruct;
    char buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "videos/bird_%Y%m%d_%H%M%S.mp4", &tstruct);
    return std::string(buf);
}

int main() {
    // 1. Setup Camera and Constants
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) return -1;

    int fps = 20;               // Lower FPS saves memory/CPU
    int preRollSecs = 2;        // Seconds to keep in RAM
    int postRollSecs = 3;       // Seconds to record after motion stops
    double motionThresh = 0.01; // 1% of pixels must move to trigger

    size_t maxBufferFrames = fps * preRollSecs;
    int postRollLimit = fps * postRollSecs;
    
    // 2. OpenCV Objects
    cv::Ptr<cv::BackgroundSubtractorMOG2> pBackSub = cv::createBackgroundSubtractorMOG2(500, 16, true);
    cv::VideoWriter writer;
    std::deque<cv::Mat> buffer;
    
    cv::Mat frame, fgMask;
    bool isRecording = false;
    int framesSinceMotion = 0;

    std::cout << "Monitoring bird feeder... Press ESC to exit." << std::endl;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // --- STEP A: Process Motion ---
        pBackSub->apply(frame, fgMask);
        // Remove shadows (gray pixels) and noise
        cv::threshold(fgMask, fgMask, 200, 255, cv::THRESH_BINARY);
        cv::erode(fgMask, fgMask, cv::Mat(), cv::Point(-1,-1), 2); 

        double motionAmount = (double)cv::countNonZero(fgMask) / fgMask.total();
        bool motionDetected = (motionAmount > motionThresh);

        // --- STEP B: Circular Buffer Management ---
        if (!isRecording) {
            buffer.push_back(frame.clone());
            if (buffer.size() > maxBufferFrames) {
                buffer.pop_front();
            }
        }

        // --- STEP C: State Machine Logic ---
        if (motionDetected) {
            framesSinceMotion = 0;
            if (!isRecording) {
                std::cout << "Motion detected! Opening file..." << std::endl;
                isRecording = true;
                
                // Use 'mp4v' for high compatibility
                std::string filename = getFilename();
                writer.open(filename, cv::VideoWriter::fourcc('m','p','4','v'), fps, frame.size());

                // DUMP PRE-ROLL BUFFER
                while (!buffer.empty()) {
                    writer.write(buffer.front());
                    buffer.pop_front();
                }
            }
        } else {
            framesSinceMotion++;
        }

        // --- STEP D: Write to Disk ---
        if (isRecording) {
            writer.write(frame);

            // Check if bird is gone (Post-roll finished)
            if (framesSinceMotion > postRollLimit) {
                std::cout << "Bird left. Saving video." << std::endl;
                isRecording = false;
                writer.release();
            }
        }

        // Optional: Show feed
        cv::imshow("BirdCam Feed", frame);
        if (cv::waitKey(1000/fps) == 27) break;
    }

    return 0;
}
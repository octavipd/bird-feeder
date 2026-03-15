#include <opencv2/opencv.hpp>
#include <deque>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace cv;
using namespace std;

// Helper to generate unique filenames
string getTimestampName() {
    auto now = chrono::system_clock::now();
    auto in_time_t = chrono::system_clock::to_time_t(now);
    stringstream ss;
    ss << "videos/bird_" << put_time(localtime(&in_time_t), "%Y%m%d_%H%M%S") << ".avi";
    return ss.str();
}

int main() {
    VideoCapture cap(2);
    if (!cap.isOpened()) return -1;

    // PS3 Eye Specific Settings (640x480 @ 60fps is the sweet spot)
    int width = 640;
    int height = 480;
    int fps = 30; 

    cap.set(CAP_PROP_FRAME_WIDTH, width);
    cap.set(CAP_PROP_FRAME_HEIGHT, height);
    cap.set(CAP_PROP_FPS, fps);

    int bufferSize = 5; 
    int postMotionDelay = 5;
    
    deque<Mat> frameBuffer;
    VideoWriter writer;
    bool isRecording = false;
    int framesSinceMotion = 0;

    Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2(500, 25, false);

    cout << "Headless Monitoring Started... Press Ctrl+C to stop." << endl;

    while (true) {
        Mat frame, fgMask;
        if (!cap.read(frame)) break;

        // 1. Motion Detection (Using a downsized mask for speed)
        Mat smallFrame;
        resize(frame, smallFrame, Size(320, 240)); 
        pBackSub->apply(smallFrame, fgMask);
        
        // Count pixels that changed
        int motionCount = countNonZero(fgMask);

        // 2. Buffer Management
        frameBuffer.push_back(frame.clone());
        if (frameBuffer.size() > bufferSize) {
            frameBuffer.pop_front();
        }

        // 3. Logic: Start Recording
        if (motionCount > 700) { // Adjust sensitivity here
            framesSinceMotion = 0;
            if (!isRecording) {
                string filename = getTimestampName();
                cout << "Motion! Saving to: " << filename << endl;
                writer.open(filename, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame.size());
                
                while (!frameBuffer.empty()) {
                    writer.write(frameBuffer.front());
                    frameBuffer.pop_front();
                }
                isRecording = true;
            }
        } else {
            framesSinceMotion++;
        }

        // 4. Logic: Continue/Stop Recording
        if (isRecording) {
            writer.write(frame);
            if (framesSinceMotion > postMotionDelay) {
                cout << "Activity ended. File closed." << endl;
                isRecording = false;
                writer.release();
            }
        }
        
        // Note: No imshow() or waitKey() here for headless operation.
    }
    return 0;
}
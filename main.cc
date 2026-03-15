#include <opencv2/opencv.hpp>
#include <deque>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

using namespace cv;
using namespace std;

// Helper to generate unique filenames
string getTimestampName()
{
        auto now = chrono::system_clock::now();
        auto in_time_t = chrono::system_clock::to_time_t(now);
        stringstream ss;
        ss << "videos/bird_" << put_time(localtime(&in_time_t), "%Y%m%d_%H%M%S") << ".avi";
        return ss.str();
}

int main()
{
        VideoCapture cap(0);
        if (!cap.isOpened())
                return -1;

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

        Ptr<BackgroundSubtractor> pBackSub = createBackgroundSubtractorMOG2(500, 25, true);

        cout << "Headless Monitoring Started... Press Ctrl+C to stop." << endl;

        int frameCounter = 0;
        int skipInterval = 4; // Process motion every 3 frames

        while (true)
        {
                Mat frame, fgMask;
                if (!cap.read(frame))
                        break;

                frameCounter++;

                // 1. Motion Detection (ONLY RUN EVERY N FRAMES)
                int motionCount = 0;
                if (frameCounter % skipInterval == 0)
                {
                        Mat smallFrame;
                        resize(frame, smallFrame, Size(160, 120));
                        pBackSub->apply(smallFrame, fgMask);

                        // Morphological Opening to remove noise (very important for false positives!)
                        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
                        morphologyEx(fgMask, fgMask, MORPH_OPEN, kernel);

                        motionCount = countNonZero(fgMask);
                }

                // 2. Buffer Management (STILL RUN EVERY FRAME)
                // We keep this outside the skip logic so the pre-roll buffer is complete.
                frameBuffer.push_back(frame.clone());
                if (frameBuffer.size() > bufferSize)
                {
                        frameBuffer.pop_front();
                }

                // 3. Logic: Start Recording
                // We only check motionCount when we actually calculated it
                if (frameCounter % skipInterval == 0)
                {
                        if (motionCount > 500)
                        {
                                framesSinceMotion = 0;
                                if (!isRecording)
                                {
                                        string filename = getTimestampName();
                                        cout << "Motion detected! Recording..." << endl;
                                        writer.open(filename, VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, frame.size());

                                        while (!frameBuffer.empty())
                                        {
                                                writer.write(frameBuffer.front());
                                                frameBuffer.pop_front();
                                        }
                                        isRecording = true;
                                }
                        }
                        else
                        {
                                // Only increment the "cooldown" timer on frames where we checked for motion
                                framesSinceMotion++;
                        }
                }

                // 4. Logic: Continue Recording (RUN EVERY FRAME)
                // This ensures the video doesn't look "choppy"
                if (isRecording)
                {
                        writer.write(frame);
                        // Since we only check framesSinceMotion every skipInterval,
                        // we should adjust postMotionDelay accordingly.
                        if (framesSinceMotion > postMotionDelay)
                        {
                                cout << "Activity ended. File closed." << endl;
                                isRecording = false;
                                writer.release();
                        }
                }
        }
        return 0;
}

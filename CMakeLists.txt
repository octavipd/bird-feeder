cmake_minimum_required(VERSION 3.10)
project(OpenCVTest)

# Find the OpenCV package on Fedora
find_package(OpenCV REQUIRED)

# Add your executable
add_executable(OpenCVTest main.cc)

# Link OpenCV libraries to your executable
target_link_libraries(OpenCVTest ${OpenCV_LIBS})
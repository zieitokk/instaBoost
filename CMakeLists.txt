cmake_minimum_required(VERSION 3.13)
project(test)

add_definitions(-std=c++11)

# 寻找OpenCV库
set(OpenCV_DIR /home/xiangtao/Downloads/opencv-3.4.0/release)
find_package( OpenCV 3 REQUIRED )

file(GLOB instaBoost ./src/*.cpp)

include_directories( ${OpenCV_INCLUDE_DIRS} )


# Add excutable target with source files listed in SOURCE_FILES variable
add_executable(test ${instaBoost})

# 链接OpenCV库
target_link_libraries( test ${OpenCV_LIBS} )

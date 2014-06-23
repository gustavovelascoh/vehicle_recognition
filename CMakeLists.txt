cmake_minimum_required(VERSION 2.8)
project( vehicle_recognition )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} include)

#add_executable( ../bin/bgsub src/bgsub.cpp )
#target_link_libraries( ../bin/bgsub ${OpenCV_LIBS} )

#add_executable( ../bin/bgfg src/bgfg_codebook.cpp )
#target_link_libraries( ../bin/bgfg ${OpenCV_LIBS} )

add_executable( ../bin/vehicle_recognition src/vehicle_recognition.cpp src/Bgsub.cpp)
target_link_libraries( ../bin/vehicle_recognition ${OpenCV_LIBS} )

add_executable( ../bin/contours src/contours.cpp )
target_link_libraries( ../bin/contours ${OpenCV_LIBS} )

add_executable( ../bin/test_contours src/test_contours.cpp )
target_link_libraries( ../bin/test_contours ${OpenCV_LIBS} )
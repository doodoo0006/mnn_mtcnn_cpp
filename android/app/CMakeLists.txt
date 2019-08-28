cmake_minimum_required(VERSION 3.4.1)

set(lib_DIR ${CMAKE_SOURCE_DIR}/libs)
set(inc_DIR ${CMAKE_SOURCE_DIR}/includes)
include_directories(${inc_DIR})
include_directories(${inc_DIR}/inc_mnn)
include_directories(${inc_DIR}/inc_opencv)

set(OpenCV_STATIC ON)
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/OpenCV-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)

add_library(native-lib SHARED src/main/cpp/native-lib.cpp )
add_library(mtcnn_demo SHARED src/main/cpp/imgProcess.cpp
                              src/main/cpp/Bbox.cpp
                              src/main/cpp/mtcnn.cpp
                              src/main/cpp/net.cpp)
                              
add_library(MNN SHARED IMPORTED )
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${lib_DIR}/${ANDROID_ABI}/libMNN.so)
target_link_libraries(mtcnn_demo MNN ${OpenCV_LIBS})

find_library(log-lib log )
target_link_libraries(native-lib mtcnn_demo MNN ${log-lib} ${OpenCV_LIBS} )
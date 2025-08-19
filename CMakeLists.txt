cmake_minimum_required(VERSION 3.14)
project(ndvi VERSION 0.1.0 LANGUAGES C CXX)

find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBRAW REQUIRED libraw)

# add_executable(ndvi main.cpp)
add_executable(ndvi main.cpp)


target_include_directories(ndvi PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${LIBRAW_INCLUDE_DIRS}
)

target_link_libraries(ndvi PRIVATE ${OpenCV_LIBS}
                                   ${LIBRAW_LIBRARIES}
)
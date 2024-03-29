CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(libNumerics)

INCLUDE_DIRECTORIES(..)

ADD_LIBRARY(Numerics
        computeH.cpp
        homography.cpp homography.h
        matrix.cpp matrix.h
        numerics.cpp numerics.h
        rodrigues.cpp rodrigues.h
        vector.cpp       
        cubicRoots.h
        ccmath_svd.cpp
)
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS "set env var")
set(ENV{CC} "/usr/local/opt/llvm/bin/clang")
set(ENV{CXX} "/usr/local/opt/llvm/bin/clang++")
set(ENV{LDFLAGS} "-L/usr/local/opt/llvm/lib")
set(ENV{CPPFLAGS} "-I/usr/local/opt/llvm/include")
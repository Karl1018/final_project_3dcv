
CMAKE_MINIMUM_REQUIRED(VERSION 3.5)
project(autosimilarities)

set(CMAKE_BUILD_TYPE Release)

FIND_PACKAGE(OpenMP)
if(OPENMP_FOUND)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ADD_SUBDIRECTORY(libNumerics)

####### Different Configs For IMAS
set(src
libautosim.cpp
library.cpp

# Minimal Algebra capabilities
libNumerics/numerics.cpp

)


add_library(autosim MODULE ${src})
if(OPENMP_FOUND)
    TARGET_LINK_LIBRARIES(autosim Numerics OpenMP::OpenMP_CXX)
else()
    TARGET_LINK_LIBRARIES(autosim Numerics)
endif()



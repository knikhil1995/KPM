##
#  CMakeList file for KPM
##
CMAKE_MINIMUM_REQUIRED(VERSION 3.17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# Set the name of the project and target:
SET(TARGET "KPM_exe")
SET(TARGETLIB "KPM")

SET(TARGET_VERSION "1.0-pre") # used in docs/Doxyfile.in

add_definitions(-w)

add_compile_options("-O2")

PROJECT(${TARGET} LANGUAGES CXX CUDA)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


ADD_EXECUTABLE(${TARGET}   src/main.cc)
set_property(TARGET "KPM_exe" PROPERTY OUTPUT_NAME "KPM")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)


# Declare all source files the target consists of:
SET(TARGET_SRC
 ./src/main.cc
  )

SET(DEVICE_SRC
  ./src/HamiltonianAnderson.cc
  ./src/moments.cc
  ./src/lanczos.cc)


set_source_files_properties(${DEVICE_SRC} PROPERTIES LANGUAGE CUDA)

LIST(APPEND TARGET_SRC ${DEVICE_SRC})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")


ADD_LIBRARY(${TARGETLIB} SHARED ${TARGET_SRC})

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

# make sure RPATHs are used on macOS
SET(CMAKE_INSTALL_NAME_DIR "${CMAKE_INSTALL_PREFIX}/lib")


#
#Set GPU compilation flag
#
find_package(CUDAToolkit REQUIRED)
target_include_directories(${TARGETLIB} PUBLIC ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${TARGETLIB} PUBLIC CUDA::cudart)
target_link_libraries(${TARGETLIB} PUBLIC CUDA::cublas)
target_link_libraries(${TARGETLIB} PUBLIC CUDA::cusparse)
target_link_libraries(${TARGETLIB} PUBLIC CUDA::cufft)

find_package(BLAS REQUIRED)
target_link_libraries(${TARGETLIB} PUBLIC ${BLAS_LIBRARIES})

find_package(LAPACK REQUIRED)
target_link_libraries(${TARGETLIB} PUBLIC ${LAPACK_LIBRARIES})

TARGET_INCLUDE_DIRECTORIES(${TARGETLIB} PUBLIC ${CMAKE_SOURCE_DIR}/include)
TARGET_LINK_LIBRARIES(${TARGET} PUBLIC ${TARGETLIB})


#
# Custom "debug" and "release" make targets:
#
ADD_CUSTOM_TARGET(debug
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)

ADD_CUSTOM_TARGET(release
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)


#
# FIXME: that will not play well with installation
#
ADD_DEFINITIONS(-DKPM_PATH="${CMAKE_CURRENT_SOURCE_DIR}")


#
# installation target.
# use with -DCMAKE_INSTALL_PREFIX=/path/to/
#
SET(TARGETLIBEXPORT "kpmTargets")
# 1. Install defined library target
install (TARGETS ${TARGETLIB} EXPORT ${TARGETLIBEXPORT}
         LIBRARY DESTINATION lib)
# 2. Install the binary
install (TARGETS ${TARGET}
         RUNTIME DESTINATION bin)
# 3. Install content of include dir
install (DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/"
         DESTINATION include)

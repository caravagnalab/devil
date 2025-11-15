cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(devil-cuda VERSION 1.0.0
		   DESCRIPTION "GPU porting of devil R package"
		   LANGUAGES CXX CUDA)
		 
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set nvcc as cuda compiler
set(CMAKE_CUDA_COMPILER nvcc)
set(CMAKE_CUDA_ARCHITECTURES "80" "70" "72" "75")

#Set flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

#Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)



#Set installation path
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
add_compile_options( -O3 -fopenmp)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Eigen3 3.3 REQUIRED)

include_directories(${CUTENSOR_HOME}/include/ ${CMAKE_SOURCE_DIR}/include ~/.local/include/ ~/.local/include/cutensor)
link_directories( ${CUTENSOR_HOME}/lib/cutensor ~/.local/lib/cutensor)

set( BETA_FIT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/inverse.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/src/einsum.cpp
			${CMAKE_CURRENT_SOURCE_DIR}/src/kernel.cu
			${CMAKE_CURRENT_SOURCE_DIR}/src/batch.cpp)

set( BETA_FIT_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/src/inverse.hpp
                        ${CMAKE_CURRENT_SOURCE_DIR}/src/einsum.hpp
                       	${CMAKE_CURRENT_SOURCE_DIR}/src/kernel.h
                       	${CMAKE_CURRENT_SOURCE_DIR}/src/batch.hpp)

set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/inverse.cpp PROPERTIES LANGUAGE CUDA)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/src/batch.cpp PROPERTIES LANGUAGE CUDA)

set(BETA_FIT_LIBRARIES  CUDA::cudart CUDA::cublas CUDA::cusolver cutensor OpenMP::OpenMP_CXX Eigen3::Eigen)

add_library(objlib OBJECT ${BETA_FIT_SOURCES})
target_link_libraries(objlib ${BETA_FIT_LIBRARIES})
set_property(TARGET objlib PROPERTY POSITION_INDEPENDENT_CODE 1)



set(BUILD_STATIC_LIBS TRUE)

if(${BUILD_STATIC_LIBS})
    set(BUILD_LIB_TARGET libBetaFit_static)
    set(LIB_INSTALL_TARGETS libBetaFit_static)
endif()

if(NOT DEFINED BUILD_SHARED_LIBS)
    set(BUILD_LIB_TARGET libBetaFit)
    set(BUILD_SHARED_LIBS TRUE)
endif()

if(${BUILD_SHARED_LIBS})
    set(LIB_INSTALL_TARGETS libBetaFit ${LIB_INSTALL_TARGETS})
endif()

if(${BUILD_SHARED_LIBS})
    add_library(libBetaFit SHARED $<TARGET_OBJECTS:objlib>)
    target_link_libraries(libBetaFit ${BETA_FIT_LIBRARIES})
    set_target_properties(libBetaFit PROPERTIES VERSION ${PROJECT_VERSION} OUTPUT_NAME BETA_FIT)
endif()

if(${BUILD_STATIC_LIBS})
    add_library(libBetaFit_static STATIC $<TARGET_OBJECTS:objlib>)
    target_link_libraries(libBetaFit_static ${BETA_FIT_LIBRARIES})
    set_target_properties(libBetaFit_static PROPERTIES VERSION ${PROJECT_VERSION} OUTPUT_NAME BETA_FIT)
endif()

#########################################

set(INSTALL_HEADERS TRUE)

if(${INSTALL_HEADERS})
  if(${BUILD_SHARED_LIBS})
    set_property(TARGET libBetaFit PROPERTY PUBLIC_HEADER ${BETA_FIT_HEADER})
  endif()

  if(${BUILD_STATIC_LIBS})
     set_property(TARGET libBetaFit_static PROPERTY PUBLIC_HEADER ${BETA_FIT_HEADER})
  endif()
  install(TARGETS ${LIB_INSTALL_TARGETS} LIBRARY
          ARCHIVE
          PUBLIC_HEADER)

else()
  install(TARGETS ${LIB_INSTALL_TARGETS})
endif()



#Compile test

add_executable(test.x src/test.cpp src/inverse.cpp)
target_link_libraries(test.x gtest_main CUDA::cudart cutensor CUDA::cublas)
enable_testing()
include(GoogleTest)
gtest_discover_tests(test.x)

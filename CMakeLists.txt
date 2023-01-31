# cmake_minimum_required(VERSION 3.3)
# project(nerfacc LANGUAGES CXX CUDA)

# find_package(pybind11 REQUIRED)
# find_package(Torch REQUIRED)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# set(SOURCE_DIR nerfacc/cuda/csrc)
# set(INCLUDE_DIR nerfacc/cuda/csrc/include)
# file(GLOB SOURCES ${SOURCE_DIR}/*.cu)

# pybind11_add_module(${PROJECT_NAME} SHARED ${SOURCES})
# target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")
# target_include_directories(${PROJECT_NAME} PRIVATE "${INCLUDE_DIR}")


# # message(STATUS "CUDA enabled")

# # set( CMAKE_CUDA_STANDARD 14 )
# # set( CMAKE_CUDA_STANDARD_REQUIRED ON)

# # find_package(pybind11 REQUIRED)

# # # find_package(Python3 REQUIRED COMPONENTS Development)
# # # target_link_libraries(${PROJECT_NAME} PRIVATE Python3::Python)

# # find_package(Torch REQUIRED)
# # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# # target_link_libraries(${PROJECT_NAME} PRIVATE ${TORCH_LIBRARIES})

# # set(CSRC nerfacc/cuda/csrc)
# # file(GLOB_RECURSE ALL_SOURCES ${ALL_SOURCES} ${CSRC}/*.cu)
# # file(GLOB_RECURSE ALL_HEADERS ${CSRC}/include/*.h)
# # add_library(${PROJECT_NAME} SHARED ${ALL_SOURCES})
# # target_include_directories(${PROJECT_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# # set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0")

# # message("-- CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
# # message("-- CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
# # message("-- CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")

# # set_target_properties(${PROJECT_NAME} PROPERTIES
# #   EXPORT_NAME nerfacc
# #   INSTALL_RPATH ${TORCH_INSTALL_PREFIX}/lib)

# # Cmake creates *.dylib by default, but python expects *.so by default
# # if (APPLE)
# #   set_property(TARGET ${PROJECT_NAME} PROPERTY SUFFIX .so)
# # endif()
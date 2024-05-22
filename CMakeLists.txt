cmake_minimum_required(VERSION 3.7)
project(ParallelReduction)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake_modules)
message(cmake_module_path: ${CMAKE_MODULE_PATH})
find_package(OpenCL REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(${OpenCL_INCLUDE_DIRS})
add_executable(${PROJECT_NAME}    main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})

add_compile_options(${PROJECT_NAME} -Wall)
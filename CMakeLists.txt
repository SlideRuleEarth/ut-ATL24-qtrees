cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(ATL24_qtrees VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(xgboost REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

set(CMAKE_CXX_FLAGS "-Wall -Werror -Wshadow ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/apps)

############################################################
# Unit tests
############################################################
macro(add_test name)
    add_executable(${name} ./tests/${name}.cpp)
    target_link_libraries(${name} xgboost::xgboost)
endmacro()

add_test(test_utils)
add_test(test_xgb1)
add_test(test_xgb2)

############################################################
# Applications
############################################################
add_executable(train ./apps/train.cpp)
target_link_libraries(train xgboost::xgboost)
target_precompile_headers(train PUBLIC apps/precompiled.h)

add_executable(classify ./apps/classify.cpp)
target_link_libraries(classify xgboost::xgboost)
target_precompile_headers(classify PUBLIC apps/precompiled.h)

add_executable(score ./apps/score.cpp)
target_link_libraries(score)
target_precompile_headers(score PUBLIC apps/precompiled.h)

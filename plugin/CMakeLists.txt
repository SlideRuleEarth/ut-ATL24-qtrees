cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(ATL24_qtrees VERSION 1.0.0 LANGUAGES CXX)

find_package(OpenMP REQUIRED)
find_package(xgboost REQUIRED)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

set(CMAKE_CXX_FLAGS "-Wall -Werror ${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(INSTALLDIR /usr/local CACHE STRING "Installation directory for library and executables")

# Plugin #
add_library(qtrees SHARED ${CMAKE_CURRENT_LIST_DIR}/qtrees.cpp)
target_sources(qtrees PRIVATE ${CMAKE_CURRENT_LIST_DIR}/QtreesClassifier.cpp)
target_link_libraries(qtrees xgboost::xgboost)
target_include_directories (qtrees PUBLIC ..)
target_include_directories (qtrees PUBLIC ../apps)
target_include_directories (qtrees PUBLIC ${INSTALLDIR}/include/sliderule)
target_include_directories (qtrees PUBLIC ${INSTALLDIR}/include/sliderule/bathy)
target_include_directories (qtrees PUBLIC ${INSTALLDIR}/include/sliderule/icesat2)
target_compile_definitions (qtrees PUBLIC BINID="v0.0.1")
target_precompile_headers(qtrees PUBLIC ../apps/precompiled.h)
set_target_properties (qtrees PROPERTIES OUTPUT_NAME qtrees)
set_target_properties (qtrees PROPERTIES PREFIX "")


# Build Information #
execute_process (COMMAND git --work-tree ${PROJECT_SOURCE_DIR}/.. --git-dir ${PROJECT_SOURCE_DIR}/../.git describe --abbrev --dirty --always --tags --long OUTPUT_VARIABLE BUILDINFO)
string(REGEX REPLACE "\n$" "" BUILDINFO "${BUILDINFO}")
target_compile_definitions (qtrees PUBLIC BUILDINFO="${BUILDINFO}")

# Plugin Installation #
install (TARGETS qtrees LIBRARY DESTINATION ${INSTALLDIR}/etc/sliderule)

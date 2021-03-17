cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(NAME 001_example)
project(${NAME})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

add_executable(${NAME} main.cpp)
set_property(TARGET ${NAME} PROPERTY CXX_STANDARD 14)


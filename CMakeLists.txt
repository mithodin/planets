cmake_minimum_required(VERSION 3.5)

FIND_PACKAGE(ArrayFire)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})

ADD_EXECUTABLE(planets planets.cpp)
TARGET_LINK_LIBRARIES(planets ${ArrayFire_LIBRARIES} )

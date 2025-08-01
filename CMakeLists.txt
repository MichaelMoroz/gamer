cmake_minimum_required(VERSION 3.10)
project(gamer_cli LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS ${CMAKE_SOURCE_DIR})

find_package(Qt5 REQUIRED COMPONENTS Core Gui Widgets OpenGL)
find_package(OpenGL REQUIRED)
find_package(OpenMP)
include_directories(${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/source ${CMAKE_SOURCE_DIR}/source/galaxy ${CMAKE_SOURCE_DIR}/source/noise ${CMAKE_SOURCE_DIR}/source/util)

set(SOURCES
    dialogabout.cpp
    dialogrendererhelp.cpp
    source/consolerenderer.cpp
    source/glwidget.cpp
    source/main.cpp
    source/mainwindow.cpp
    source/openglwindow.cpp
    source/trianglewindow.cpp
    source/window.cpp
    source/galaxy/componentparams.cpp
    source/galaxy/galaxy.cpp
    source/galaxy/galaxycomponent.cpp
    source/galaxy/galaxycomponents.cpp
    source/galaxy/galaxyinstance.cpp
    source/galaxy/galaxyparams.cpp
    source/galaxy/gamercamera.cpp
    source/galaxy/hpxrasterizer.cpp
    source/galaxy/rasterizer.cpp
    source/galaxy/rasterizeromp.cpp
    source/galaxy/rasterpixel.cpp
    source/galaxy/rasterthread.cpp
    source/galaxy/renderingparams.cpp
    source/galaxy/renderqueue.cpp
    source/galaxy/spectrum.cpp
    source/noise/iqnoise.cpp
    source/noise/noise.cpp
    source/noise/perlin.cpp
    source/noise/simplex.cpp
    source/noise/simplexnoise.cpp
    source/util/buffer2d.cpp
    source/util/cmpi.cpp
    source/util/fitsio.cpp
    source/util/fitsparam.cpp
    source/util/gmessages.cpp
    source/util/util.cpp
)

set(RESOURCES gamerresources.qrc)
set(UIS mainwindow.ui dialogrendererhelp.ui dialogabout.ui)

add_executable(gamer_cli ${SOURCES} ${RESOURCES} ${UIS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(gamer_cli PRIVATE OpenMP::OpenMP_CXX)
endif()

target_link_libraries(gamer_cli PRIVATE Qt5::Core Qt5::Gui Qt5::Widgets Qt5::OpenGL OpenGL::GL)


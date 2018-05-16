#in questa versione trova automaticamente vtk9
#ma non trova vtkActor (vedi build5)
PROJECT(FindBorder)
cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
find_package(VTK REQUIRED)
# include(${VTK_USE_FILE}) #con questo non compila la libreria STARAlgorithm
include_directories(${VTK_INCLUDE_DIRS})
include_directories(src)

add_subdirectory(src)
 
add_executable(FindBorder Demo.cxx )


if(VTK_LIBRARIES)
  set(LIBS ${LIBS} ${VTK_LIBRARIES})
else()
 target_link_libraries(FindBorder vtkHybrid vtkWidgets)
endif()

set(LIBS ${LIBS} STARAlgorithm)
target_link_libraries(FindBorder ${LIBS})

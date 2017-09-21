cmake_minimum_required(VERSION 2.8.5 FATAL_ERROR)
 
PROJECT(FindBorder)
 
find_package(VTK REQUIRED)
include(${VTK_USE_FILE})
 
 add_executable(FindBorder MACOSX_BUNDLE Demo.cxx )


if(VTK_LIBRARIES)
  target_link_libraries(FindBorder ${VTK_LIBRARIES}  -L./src  -ldl -Wl,-rpath,./src -ltest -L/usr/local/cuda/lib64 -lcudart)
else()
  target_link_libraries(FindBorder vtkHybrid vtkWidgets)
endif()

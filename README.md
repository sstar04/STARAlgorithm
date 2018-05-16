# STARAlgorithm
The STAR (Shape of Tumor Algorithmic Reconstruction) algorithm defines the shape of cluster of cells and searches for the boudary and voids in 3D environment. The algorithm is GPU based and it is written using CUDA programming interface by NVIDIA and the Thrust library. The code include a function for visualization purpose implemented using VTK library.

Code included in Thrust and CUDA example was used for the implementation of some step of the algorithm. You can find details in CellSearch.cu file.

Prerequisites
=============
NVIDIA Graphic card with CUDA support

VTK library


Linux instruction for running demo 
===========
1. Copy STARAlgortihm anywhere you like and enter the folder

     cd STARAlgorithm

2. Use CMakeLists.txt to compile the code sample Demo.cpp 

     mkdir build
     
     cd build 
     
     ccmake .. 
     
      (If VTK is not installed but compiled on your system, you will need to specify the path to your VTK build
      (see https://www.vtk.org/Wiki/VTK/Configure_and_Build for more details)
       
     make

4. Run the executable FindBorder

   Some data set examples are available in folder DataSetExample, move the folder in the directory where you are running the code.
   
   The folder DataSetExamples contains vtk files named "Cells#.vtk" storing space coordinates and radius of the cells composing a simulated tumor spheroid.
   
   You can use the number of the file to select the dataset you want to analize. 

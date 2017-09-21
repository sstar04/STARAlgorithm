# STARAlgorithm
The STAR (Shape of Tumor Algorithmic Reconstruction) algorithm defines the shape of cluster of cells and searches for the boudary and voids in 3D environment. The algorithm is GPU based and it is written using CUDA programming interface by NVIDIA and the Thrust library. The code include a function for visualization purpose implemented using VTK library.

Prerequisites
=============
NVIDIA Graphic card with CUDA support
VTK library


Linux instruction for running demo 
===========
1. Copy TestAlgortihm anywhere you like
     cd TestAlgorithm

2. Use the makefile_lib to compile the CUDA code  
     make -f src/makefile_lib

3. Compile Demo.cxx using cmake
     ccmake . 
      (If VTK is not installed but compiled on your system, you will need to specify the path to your VTK build
       (see https://www.vtk.org/Wiki/VTK/Configure_and_Build for more details)
     make

4. Run the executable FindBorder

   Some data set examples are available in folder DataSetExample. They consist in vtk files named "Cells#.vtk" which contains space coordinates and radius of the cells composing a simulated tumor spheroid.
   You can use the number of the file to select the dataset you want to analize. 
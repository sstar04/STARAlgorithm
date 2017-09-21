
/*
 * CellSearch.cu
 * In this file the function CellSearch is implemented. It executes the algorithm illustrate in fig4. in the paper.
 * CellSearch calls external function for the parallel execution of the connected component labeling algorithm required to find
 * the subgraphs, and it also performs kernel calls which are defined in DataStructure.h and GridConstruction.cuh files.
 *
 *
 *
 *  Created on: 20/set/2017
 *      Author: Sabrina Stella
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "macro.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/unique.h>
#include <thrust/functional.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>
#include <thrust/transform_reduce.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/sequence.h>
#include <thrust/replace.h>
#include <thrust/distance.h>
#include <cmath>
#include "DataStructure.h"
#include "GridConstruction.cuh"
#include "CCL_Algorithm.cuh"
#include "StarAlg_kernel.cuh"
#include <algorithm>
#include <thrust/copy.h>
#include "CellSearch.h"

using namespace std;

//Print execution time flag
bool printExecTime = true;

//debugging flags
bool debugAdCell = false;
bool scaleNumber = false;
bool checkBorder=false;

//visualization flags
bool showStatusNodes = false;
bool showBorderNodes = false;
bool showLabelNodes = false;


// external function implemented in "CCL_Algorithm.cuh"
extern void CCL_Algorithm_2D(
		thrust::device_vector<uint>  &node_state,              //input data
		thrust::device_vector<uint>& nodes_label,              //output data for visualize result
		//	vector<uint>& border_cell                          //output data for find the border cell.
		uint size, ofstream& ofs);


////////
/// This function reads - form an external file - the point set NPART and return the list of boundary points
/// NPART = total number of 4D points: (3D coordinates,radius):= (x,y,z,r)
/// NCELLS = total number of partition cubes

void CellSearch( vector<uint>& nodes_label, Params4Visual* P)
{

    //////
	//read 4D points from a vtk file
	//////

	ifstream fileCoord, fileData;
    char fileName[30];
    int num;
    cout<<"insert file number ";
    if(scaleNumber)cout <<"(position expressed in meter)";
    cin>>num;
    sprintf (fileName, "DataSetExamples/Cells%d.vtk", num);
    cout << fileName << endl;

    fileCoord.open(fileName);
    string line;
    string a, b, c;
    for(int i=0; i<4 ; i++)getline(fileCoord,line);
    fileCoord >> a >> b >> c;
    int NPART = atoi(b.c_str());
    cout << "Npart:" << NPART << endl;

    //ponter on radius value
    fileData.open(fileName);
    string line2, r;
    string word("radius");
    std::size_t found;

    do{
    	getline(fileData,line2);
    	found = line2.find(word);
    }while(found==std::string::npos);
    getline(fileData,line2);
    fileData >> r;

    //open output file
    char outputFileName[30];
    sprintf (outputFileName, "DataSetExamples/times%d.txt", num);
    std::ofstream ofs (outputFileName, std::ofstream::out);
    ofs << "OUTPUT FILE " << fileName << endl;
    ofs << "Npart:" << NPART << endl;

    //allocate 4D data on a device_vector

    thrust::device_vector<point4d> points(NPART);
    fileCoord >> a >> b >> c;
    int i = 0;
    while(a !="POINT_DATA"){
    	 if(scaleNumber){ points[i]=point4d(atof(a.c_str())*pow(10,-6),atof(b.c_str())*pow(10,-6),atof(c.c_str())*pow(10,-6),atof(r.c_str())*pow(10,-6));}
    	 else points[i]=point4d(atof(a.c_str()),atof(b.c_str()),atof(c.c_str()),atof(r.c_str()));
    	 if (checkGrid) cout << "( " << atof(a.c_str()) << ", " << atof(b.c_str()) <<  ", " << atof(c.c_str()) << ", " << atof(r.c_str()) << ")" << endl;
    	 fileCoord >> a >> b >> c;
    	 fileData >> r;
    	 i++;
    }



    //////////
    // Space subdivision
    //////////

    //1. Defining the bounding box
    // This step is implemented using thrust library using the code provided in thrust examples at
    // https://github.com/thrust/thrust/blob/master/examples/bounding_box.cu

    cudaEvent_t start, stop;  // create cuda event handles
    CUDA_CALL(cudaEventCreate(&start));
    CUDA_CALL(cudaEventCreate(&stop));
    
    //initial bounding box contain first point
    bbox init = bbox(points[0],points[0]);

    //transform operations
    bbox_transform unary_op;

    //binary reduction
    bbox_reduction binary_op;

    float bbox_time = 0.0f;
    
    //start record time
    cudaEventRecord(start, 0);
    bbox result = thrust::transform_reduce(points.begin(), points.end(), unary_op, init, binary_op);
    cudaDeviceSynchronize();
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CALL(cudaEventElapsedTime(&bbox_time, start, stop));

    if(printExecTime) cout <<"bounding box execution time: " << bbox_time << endl;

    //print output;
    if (checkGrid){
    cout << "Exact bounding box" << fixed;
    cout << "(" << result.first.x << "," << result.first.y << "," << result.first.z << ")" ;
    cout << "(" << result.second.x << "," << result.second.y << "," << result.second.z  <<")" <<endl;
    }

    //2. Define the extended bounding box

    //set the dimension of the  single small cubic box
    float alfa;                //default value 2;
    float r_max = 8;          //maximum value for the cell radius

    cout << "Set the value for the space partitioning parameter (alfa*r_max)(defult: 2)";
    cin >> alfa;
    cout << "the side of the small cubic box is: " <<alfa*r_max << "micron" << endl;

    globalParams.cellSide = (alfa*r_max);

    bbox box3D;
    box3D = Bbox_EnlargeDivide(result);  //set two layes of extra space between cluser and cubic boundary

    if (checkGrid){
    	cout << "Extended bounding box" << fixed;
    	cout << "(" << box3D.first.x << "," << box3D.first.y << "," << box3D.first.z << ")" ;
    	cout << "(" <<  box3D.second.x << "," << box3D.second.y << "," << box3D.second.z  <<")" <<endl;

    	cout << "the side of the bounding box is large: " << globalParams.bboxSide << endl;
    	cout << "the vector for the origin translation is: " << globalParams.originTranslation.x << " " << globalParams.originTranslation.y << " " <<  globalParams.originTranslation.z << endl;
    }

    //inizialize a variable on the GPU's constant memory
    cudaMemcpyToSymbol(params, &globalParams, sizeof(SimParams));

    uint NCELLS = globalParams.NtotCell;
    cout << "Total number of space partitioning cubes: " << NCELLS << endl; // print total number of small cubic boxes
    ofs << "Ncells: " << NCELLS << std::endl;



    /////////
    //Setting a hash table and memory rearrangement of the 4D data points.
    ////////
    //////// In this step the kernels implementation are copied by CUDA samples, available at CUDA toolkit package.
    
    thrust::device_vector<unsigned int> gridParticleHash(NPART);
    thrust::device_vector<unsigned int> gridParticleIndex(NPART);
    
    //wrap thrust pointer
    point4d * pos = thrust::raw_pointer_cast(&points[0]);
    unsigned int * id_cell = thrust::raw_pointer_cast(&gridParticleHash[0]);
    unsigned int * id_particle = thrust::raw_pointer_cast(&gridParticleIndex[0]);
    
   
    float calcHashAndSort_time = 0.0f;
    
    //set the execution configuration
    uint nThreads, nBlocks;
    computeGridSize(NPART, 64, nBlocks, nThreads);

    dim3 numThreads(64);
    float PP = sqrt(nBlocks);
    dim3 numBlocks(PP+1 , PP+1);

    cudaEventRecord(start, 0);  //start record time

    //calculate grid hash value for each particle
    calcHashD<<<numBlocks,numThreads>>>(id_cell, id_particle, pos, NPART);

    thrust::sort_by_key(gridParticleHash.begin(), gridParticleHash.end(), gridParticleIndex.begin());

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CALL(cudaEventElapsedTime(&calcHashAndSort_time, start, stop));
    cudaDeviceSynchronize();

    if(printExecTime) cout <<"Execution time for: hash table and particle index rearrangement" << calcHashAndSort_time << endl ;

    thrust::device_vector<uint> cellStart(NCELLS);
    thrust::device_vector<uint> cellStop(NCELLS);
    thrust::device_vector<point4d> sortedPoints(NPART);
    thrust::device_vector<float> sortedPointsCoord(NPART*3); //vettore che mi permette di savare sul file eserno i risultati
    thrust::device_vector<float> sortedRadius(NPART);

    uint * cStart = thrust::raw_pointer_cast(&cellStart[0]);
    uint * cStop  = thrust::raw_pointer_cast(&cellStop[0]);
    point4d * sPoints  = thrust::raw_pointer_cast(&sortedPoints[0]);
    float   * sCoord   = thrust::raw_pointer_cast(&sortedPointsCoord[0]);
    float   * sRadius   = thrust::raw_pointer_cast(&sortedRadius[0]);

    float ReordingMemory_time = 0.0f;

    cudaEventRecord(start, 0);    //start record time

    //rearrangement of 4D points vector according to the new particle index
    reorderDataAndFindCellStartD<<<numBlocks,numThreads>>>(cStart, cStop, sPoints, id_cell, id_particle, pos, NPART);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CALL(cudaEventElapsedTime(&ReordingMemory_time, start, stop));

    if(printExecTime)std::cout <<"Execution time for: point data memory rearrangement" << ReordingMemory_time << std::endl ;

    transferPointCoord<<<numBlocks,numThreads>>>(sPoints, sCoord, sRadius, NPART);

    ///////////
    //calculate cellsize (: the total number of particle for a single small cube)
    ///////////
    thrust::device_vector<uint> cellSize(NCELLS);
    uint * cSize  = thrust::raw_pointer_cast(&cellSize[0]);
    thrust::transform(cellStop.begin(), cellStop.end(), cellStart.begin(), cellSize.begin(),
    		thrust::minus<uint>());
    if (checkGrid){
    	std::cout << "\t" << i << " : ( Start index, end index ) " << std::endl;
    	for (int i=0 ; i<NCELLS ; i++){
    		std::cout << "hash number " << i << " : (" <<  cellStart[i] <<  "," << cellStop[i] << ")" << " " << cellSize[i] << std::endl;
    	}
    }



    ///////////
    // Boundary search
    ///////////

    thrust::device_vector<bool> borderCell(NPART);
    bool * bCell  = thrust::raw_pointer_cast(&borderCell[0]);

    thrust::device_vector<uint> node_state(NCELLS);
    thrust::device_vector<uint> node_label_dev(NCELLS);
    uint * n_state  = thrust::raw_pointer_cast(&node_state[0]);
    uint * n_label_dev  = thrust::raw_pointer_cast(&node_label_dev[0]);

    //initialize node property vector
    set_number setProperty;
    thrust::transform_if(thrust::cuda::par, cellSize.begin(), cellSize.end(),node_state.begin(), setProperty , is_positive());
       
    //connected component labeling algorithm
    CCL_Algorithm_2D(node_state, node_label_dev, globalParams.s, ofs);

    P->s = globalParams.s;
    P->NtotCell = globalParams.NtotCell;


    /////////
    //Detect subgraphs pointing to the environment
    ////////

    nodes_label.resize(NCELLS); //host vector to store the output results

    for (int i =0 ; i< NCELLS ; i++){
    	nodes_label[i] = node_label_dev[i];
    }

    //set the execution configuration
    //
    uint nThreads_new, nBlocks_new;
    computeGridSize(NCELLS, 64, nBlocks_new, nThreads_new);

    dim3 numThreads_new(8,8);  //dimensione del singolo blocco
    float PP2 = sqrt(nBlocks_new);
    dim3 numBlocks_new(PP2+1 , PP2+1);


    // 1. set the list of ID-subgraphs having nodes at the bounding box
    thrust::device_vector<uint> list_border_label(NCELLS);
    uint* list_bl = thrust::raw_pointer_cast(&list_border_label[0]);

    //IntIterator newEnd0 = thrust::remove(thrust::device ,list_border_label.begin() , list_border_label.end(),-1);
    float findEnvSub_time = 0.0;
    cudaEventRecord(start, 0);

    // return the list of label subgraph that belong to the environment
    ListBorderLabelSubgraph<<<numBlocks_new, numThreads_new>>> (n_state, n_label_dev, globalParams.s, list_bl);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&findEnvSub_time, start, stop);

    if(printExecTime){
    	std::cout <<" **************" << std::endl ;
    	std::cout <<"Execution time for the environment subgraph search: " << findEnvSub_time << std::endl ;
    	std::cout <<" **************" << std::endl ;
    }
    ofs <<"Execution time for the environment subgraph search: " << findEnvSub_time << std::endl ;

    if(checkBorder){
    	cout << " hash value, list_label , list_border_label, node_state" << endl;
    	for(int i = 0 ; i<NCELLS; i++ ){
    		cout << i << " " << nodes_label[i] << list_border_label[i] << " "  << node_state[i] <<  endl;
    	}
    }

    //Eliminate -1 value from the list_border_hash vector.
    int nullvalue=NCELLS+1;
    IntIterator newEnd = thrust::remove(thrust::device ,list_border_label.begin() , list_border_label.end(),nullvalue);

    if(checkBorder){
    	cout << "list of subgraph indexes" << endl;
    	for(IntIterator i = list_border_label.begin(); i < newEnd; i++){
    		cout << *i << ", ";}
    	cout << endl;
    }

    //delete recurrent value from the list
    IntIterator newEnd_label = thrust::unique(thrust::device, list_border_label.begin(), list_border_label.end());


    //2: set node_state = 2 to all nodes of the previously selected subgraph
    // STEP2: propago lo stato environment a tutti i nodi  dei sottografi di list_border_label  //
        int Nlabels = thrust::distance(list_border_label.begin(),newEnd_label);



    float setStatus2_time = 0.0;
    cudaEventRecord(start, 0);

    SetEnvironmentState<<<numBlocks_new, numThreads_new>>>(list_bl, Nlabels, n_label_dev, globalParams.s ,n_state);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&setStatus2_time, start, stop);

    if(printExecTime){
    	cout <<" **************" << endl;
    	cout <<"Execution time for broadcasting the value 2: " << setStatus2_time << std::endl ;
    	cout <<" **************" << endl;
    }

    ofs <<" Execution time for broadcasting the value 2 at the environment subgraphs: " << setStatus2_time << std::endl ;


    ///3: Detect border nodes
    // Search for "full" nodes  having at least a single neighbouring "environment" node
    thrust::device_vector<int> list_border_nodes(NCELLS);
    int* list_bn = thrust::raw_pointer_cast(&list_border_nodes[0]);

    thrust::device_vector<uint> hash_value(NCELLS);
    uint * h_value  = thrust::raw_pointer_cast(&hash_value[0]);

    //fill the vector of hash table.
    thrust::sequence(thrust::cuda::par,hash_value.begin(), hash_value.end());

    float findBorderNodes_time = 0.0;
    cudaEventRecord(start, 0);
    Detect_BorderNodes<<<numBlocks_new, numThreads_new>>> (n_state, h_value,  NCELLS, list_bn);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&findBorderNodes_time, start, stop);
    if(printExecTime){
    	std::cout <<" **************" << std::endl ;
    	std::cout <<" Execution time for border node detection : " << findBorderNodes_time << std::endl ;
    	std::cout <<" **************" << std::endl ;
    }
    ofs <<" Execution time for border node detection: " << findBorderNodes_time << std::endl ;


    ///////////////////
    //SET VISUALIZATION
    cout << "set: 2 to visualize subgraphs;  1 to visualize node properties: empty, fill, environment; 0 to visualize boundary nodes ";
    int kkk = 0;
    cin >> kkk;
    switch(kkk)
    {
    case(2):
            		  showLabelNodes = true;
    break;
    case(1):
        			  showStatusNodes = true;
    break;
    default:
    	              showBorderNodes = true;
    }
    ///////////////////

    // Transfer visualization data in the host node_label vector
    if(showStatusNodes){
    	for(int i = 0 ; i<NCELLS; i++ ){
    		cout << i << " " << nodes_label[i] << " "  << node_state[i] <<  endl;
    		nodes_label[i] = node_state[i];
    	}
    	cout << "Node property visualization: red = full: black = empty; green = environment" << endl;
    }

    if(showLabelNodes){
    	/*
    	for(int i = 0 ; i<NCELLS; i++ ){
    		if(nodes_label[i] != 0){
    			cout << i << " " << nodes_label[i] << endl;
    		}
    		cout << i << " " << nodes_label[i] << endl;
    	}*/
    	cout << "Connected subgraphs visualization" << endl;
    }

    if(showBorderNodes){
    	for(int i = 0 ; i<NCELLS; i++ ){
    		if(list_border_nodes[i] != -1) nodes_label[i] = 1; //border node in red
    		else nodes_label[i] = 0; //set black the remaining nodes
    	}
    	cout << "Border node visualization: red = border node; black = the remaining nodes" << endl;
    }

    ////////////
    //LIST of point data at the boundary of the cluster of cells
    ///////////
    //Select those 4D data points included in the border nodes
    ///////////
    thrust::fill(thrust::cuda::par,borderCell.begin(), borderCell.end(), false);
    float findBorderCells_time = 0.0;
    cudaEventRecord(start, 0);

    Find_ListBorderCells<<<numBlocks_new, numThreads_new>>>(sPoints,cStart,cStop,list_bn, bCell, NCELLS);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&findBorderCells_time, start, stop);

    if(printExecTime){
    std::cout <<" **************" << std::endl ;
    std::cout <<" Execution time to determine the list of point data at the boundary of the cell cluster: " << findBorderCells_time << std::endl ;
    std::cout <<" **************" << std::endl ;
    }

    ofs <<" Execution time to determine the list of point data at the boundary of the cell cluster : " << findBorderCells_time << std::endl ;
    if(checkBorder){
    	for(int i = 0 ; i<borderCell.size(); i++ ){
    		cout << i << " = " << borderCell[i] << endl;
    	}
    }

    ofs.close();
    
     return;
}



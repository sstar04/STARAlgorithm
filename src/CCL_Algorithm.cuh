/*
 * Algorithm.h
 *
 *  Created on: 10/mar/2015
 *      Author: Sabrina Stella
 */
#ifndef ALGORITHM_CUH_
#define ALGORITHM_CUH_

#include <iostream>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/replace.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include "DataStructure.h"
#include "CCL_Algorithm_kernel.cuh"

using namespace std;

//timing flags
bool showTime = false;

//debuggung flag
bool debug = false;

typedef  thrust::device_vector< uint >::iterator  IntIterator;

//////
//Host functions declaration
//////
uint Renumber_subgraph(thrust::device_vector<uint>  &v, uint NCELLS);
//uint Get_CorrespondingValue(thrust::device_vector<uint>& label_copy, IntIterator& newEnd, int oldlabel);


// GPU implementation for the connected component labeling algorithm
void CCL_Algorithm_2D( thrust::device_vector<uint>  &node_state, thrust::device_vector<uint>  &node_label , uint s, ofstream& ofs){

	uint NCELLS = s*s*s;

    if(debug){
	cout << "CCLAlgorithm" << endl;
	for(int i=0 ; i<NCELLS ; i++)
		{
			cout << "cella " << i << "e' di tipo: " << node_state[i] << " "<< endl;
		}
	}
	uint * n_state  = thrust::raw_pointer_cast(&node_state[0]);
	uint * n_label  = thrust::raw_pointer_cast(&node_label[0]);

	thrust::device_vector<uint> hash_value(NCELLS);
	uint * h_value  = thrust::raw_pointer_cast(&hash_value[0]);

	//fill the node label with the corresponding hash value.
	//assign the value i at the node_label[i] position.
    thrust::sequence(thrust::cuda::par,node_label.begin(), node_label.end());
    hash_value = node_label;

    bool m;  //host variable to control for loop
    bool *m_d;
    cudaMalloc(&m_d, sizeof(bool));

    //set the execution configuration
    dim3 numThreads_new(8,8);  //single block dimension
    uint threadsblock = 64;

    uint stride = s*s;
    uint nBlocks_new = iDivUp(stride,threadsblock);
    int PP2 = ceil(sqrt(nBlocks_new));
    dim3 numBlocks_new(PP2 , PP2);

    //start record time
    float kernelB_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    int loop = 0;
    do{
    	cudaMemset(m_d, false, sizeof(bool));

    	Mesh_Kernel_B<<<numBlocks_new, numThreads_new>>>(n_state, n_label, NCELLS, m_d, loop);

    	cudaMemcpy(&m, m_d, sizeof(bool), cudaMemcpyDeviceToHost);
        loop++;
    	if(debug)cout << "m = " << m << endl;

    }while(m);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CALL(cudaEventElapsedTime(&kernelB_time, start, stop));

    if(showTime){
    	cout << "***********" << endl ;
    	cout <<"Execution time for CCL algorithm:" << kernelB_time << endl;
    	cout << "***********" << endl ;
    	cout << "Total number of loops: " << loop  << endl;
    }

    ofs <<"Execution time for CCL algorithm (using kernelB) :" << kernelB_time << endl ;
    ofs << "Total number of loops: " << loop  << endl;


     ////////////////////////
     // Renumber progressively the subgraph indexes
     ////////////////////////
    uint tot_subgraph;
    float RenumberS_time = 0.0f;

    cudaEventRecord(start, 0);

    tot_subgraph = Renumber_subgraph(node_label,NCELLS);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    CUDA_CALL(cudaEventElapsedTime(&RenumberS_time, start, stop));

    if(showTime){
    	cout << "***********" << endl ;
    	cout <<"Execution time for Renumber_subgraph:" << RenumberS_time << endl ;
    	cout << "***********" << endl ;
    	cout << endl  << "Total number of subgraphs "<< tot_subgraph << endl;
    }

    ofs << "Execution time for Renumber_subgraph:" << RenumberS_time << endl;
    ofs << endl  << "Total number of subgraphs "<< tot_subgraph << endl;

    cudaFree(m_d);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return;

}


// Renumber progressively the subgraph indexes
uint Renumber_subgraph(thrust::device_vector<uint>  &node_label, uint NCELLS){

	thrust::device_vector<uint> label_copy(NCELLS);
	label_copy = node_label;
	thrust::sort(label_copy.begin(), label_copy.end());

	//remove duplicate indexes
	IntIterator newEnd = thrust::unique(label_copy.begin(), label_copy.end());

	int tot_subgraph;

	// re-labelling if there exist at least two differents subgraphs
	if (thrust::distance(label_copy.begin(), newEnd) > 1 ){
		thrust::device_vector<uint> node_label_temp(NCELLS);
		node_label_temp = node_label;

		//re-labelling the node_label vector
		uint new_label, old_label;
		new_label= 0;

		for (IntIterator i = label_copy.begin(); i < newEnd; i++){
			old_label= *i;
			thrust::replace_copy(thrust::device, node_label_temp.begin(), node_label_temp.end(), node_label.begin(), old_label, new_label );
			node_label_temp = node_label;
			new_label++;
		}
		tot_subgraph = thrust::distance(label_copy.begin(), newEnd);
	}

	else {
		thrust::fill(thrust::device, node_label.begin(), node_label.end(), 0);
		tot_subgraph = 1;
	}

	return tot_subgraph; //total number of subgraphs; subgraphs enumeration start from zero.
}


#endif

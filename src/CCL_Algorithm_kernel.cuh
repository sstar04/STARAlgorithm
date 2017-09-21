////////////////////////////////////
//Implements kernels useful perform connected component labeling algorithm
////////////////////////////////////
#include <vector_types.h>

enum {
//	setnull,
	environment,  //vale 0
	spheroid,     // vale 1
	internal      //vale 2

};

struct is_odd
{
__host__ __device__
bool operator()(int x)
{
return x % 2;
}
};
//
//Predicate
struct is_positive
{
	__host__ __device__
	bool operator()(uint x)
	{
		if(x>0)
		  return true;  // se x e' zero mi ritona 0 quindi e' un valore falso
		else return false;
	}

};

//unary functor
//convert a point to a bbox containing that point (point) ->(point,point)
//unary functor
struct set_number : public thrust::unary_function<uint &, uint>
{
	__host__ __device__
	uint operator()(uint &x)
	{
		return 1;
	}

};


//////
//device kernels implementation
/////



//debugging flags
__device__ bool check_MK=false;
__device__ bool checkk=false;

__device__
uint3 CalcGridPosition(uint index)
{
	uint3 coord;
	coord.x = index/(params.s*params.s);
	coord.y = (index - coord.x * (params.s*params.s))/params.s;
	coord.z = (index - coord.x * (params.s*params.s) - coord.y * params.s);

	return coord;
}
/*
//controllo se un dato indice del thread punta ad un elemento di bordo.
__device__
bool CheckBorderData(  uint* hash_value,  //input data
		                     uint index)      //thread index

{
	//uint* Pointer = &n_id[0]; //posizione nel vettore n_id

	uint hash = hash_value[index];

	//lista delle posizioni adiacenti ad index in una struttura ipercubica
    uint3 coord = CalcGridPosition(hash);  //convert linear index to 3d coordinate to adjust the index to the border.

    int top = hash + 1;
    int bottom = hash -1;
    if ( coord.z == (params.s-1)) { top = -1 ; return true;}
    if ( coord.z ==  0)           { bottom = -1 ; return true;}

    int right = hash + params.s;
    int left = hash - params.s;
    if ( coord.y == (params.s -1) )  {right = -1 ; return true;}
    if ( coord.y == 0 )               {   left = -1 ; return true;}

    int forward = hash +  (params.s*params.s);
    int backward = hash - (params.s*params.s);
    if (coord.x == (params.s - 1) ) {forward = -1 ;return true;}
    if (coord.x == 0) { backward = -1 ; return true;}

	return false;

}

*/

__device__
void FindListAdjacents(uint* node_state, //input data
					   uint* hash_value,  //input data
		               uint index,      //thread index
		               int* n_id)     //output data
{
	//uint* Pointer = &n_id[0]; //posizione nel vettore n_id

	uint hash = hash_value[index];

	//lista delle posizioni adiacenti ad index in una struttura ipercubica
    uint3 coord = CalcGridPosition(hash);  //convert linear index to 3d coordinate to adjust the index to the border.

    int top = hash + 1;
    int bottom = hash -1;
    if ( coord.z == (params.s-1)) top = -1 ;
    if ( coord.z ==  0)           bottom = -1 ;

    int right = hash + params.s;
    int left = hash - params.s;
    if ( coord.y == (params.s -1) ) right = -1 ;
    if ( coord.y == 0 )                  left = -1 ;

    int forward = hash +  (params.s*params.s);
    int backward = hash - (params.s*params.s);
    if (coord.x == (params.s - 1) ) forward = -1 ;
    if (coord.x == 0) backward = -1 ;

	int k[6] = {top, bottom, right, left, forward, backward}; //indice temporaneo del vicino

	if (check_MK){
		if (index == 4) {
			printf("Stampo vettore k \n");
			for (int i = 0; i < 6 ; i++) printf("%d, ", k[i] );
		}
	}
	//lista degli adiacenti con lo stesso stato
	for (int i = 0; i < 6 ; i++){

		if( k[i] != -1 &&  node_state[k[i]] == node_state[index]) {
			n_id[i] = k[i] ;
			//ii++; // next int position
		}
		else n_id[i] = -1 ;
	}


}


__device__
void FindListAdjacentsOnly(  uint* hash_value,  //input data
		                     uint index,      //thread index
		                      int* n_id)     //output data
{
	//uint* Pointer = &n_id[0]; //posizione nel vettore n_id

	uint hash = hash_value[index];

	//lista delle posizioni adiacenti ad index in una struttura ipercubica
    uint3 coord = CalcGridPosition(hash);  //convert linear index to 3d coordinate to adjust the index to the border.

    int top = hash + 1;
    int bottom = hash -1;
    if ( coord.z == (params.s-1)) top = -1 ;
    if ( coord.z ==  0)           bottom = -1 ;

    int right = hash + params.s;
    int left = hash - params.s;
    if ( coord.y == (params.s -1) ) right = -1 ;
    if ( coord.y == 0 )                  left = -1 ;

    int forward = hash +  (params.s*params.s);
    int backward = hash - (params.s*params.s);
    if (coord.x == (params.s - 1) ) forward = -1 ;
    if (coord.x == 0) backward = -1 ;

	int k[6] = {top, bottom, right, left, forward, backward}; //indice temporaneo del vicino

	for (int i = 0; i < 6 ; i++) n_id[i] = k[i];

}

//////
//global kernels implementation
/////


__global__

void Mesh_Kernel_A(uint* node_state,
		           uint* hash_value,  //input value
		           uint* node_label,  //output vector
		           bool* m_d,         // variable
		           uint NCELLS)

{
	uint index;  // thread index

	uint x = threadIdx.x + blockIdx.x * blockDim.x ;  //faster index
	uint y = threadIdx.y + blockIdx.y * blockDim.y ;
	index = x + y * (blockDim.x * gridDim.x);

	//assign a thread for each single  cell
	if (index < NCELLS)
	{

		int n_id[6] ; //vettore dove annotare l'indice dei vicini ipercubici con lo stesso stato per ogni nodo
		uint label = node_label[index];

        //find the list of adjacent with the same state: fill the vector n_id
		FindListAdjacents(node_state, hash_value, index, n_id);

		//loop over n_id to check the lower index
		for(int i=0; i<6; i++){

		   if(n_id[i] != -1 && node_label[n_id[i]] < label){

			   label=node_label[n_id[i]];
			   *m_d = true;
		   }
		   else n_id[i] = -1 ;
		} //end for

		__syncthreads();
		node_label[index] = label;
	} /* end if on thread index */
}


__global__
void Mesh_Kernel_B(uint* node_state,   //input value
		uint* node_label,  		      //output vector
		uint NCELLS,
		bool* m_d,
		int loop)


{

#define BDIMX   8  // tile (and threadblock) size in x
#define BDIMY   8  // tile (and threadblock) size in y
#define radius  1  // 2 stencil (k/2)

	__shared__ int s_data_label[BDIMY+2*radius][BDIMX+2*radius];
	__shared__ int s_data_state[BDIMY+2*radius][BDIMX+2*radius]; //potrei ridurli ad un unico tile di tipo int2

	uint x = threadIdx.x + blockIdx.x * blockDim.x ;
	uint y = threadIdx.y + blockIdx.y * blockDim.y ;

	uint index = x + y * params.s;  // params.s is the size of the bounding box.
	int stride = params.s*params.s;  //to move along z axis

	if(x < params.s && y < params.s) { //select those threads that correspond to a single subdivision cube

		//copy data (node index and property) for the stencil at the position (x,y)
		int infrontstate, behindstate;           // data along z direction
		int infrontlabel, behindlabel;
		uint currentlabel = node_label[index];  //current thread
		uint currentstate = node_state[index];

		int tx = threadIdx.x + radius; //thread index in shared memory tale (considering the halo correction)
		int ty = threadIdx.y + radius;

		//fill the z axis data
		int out_index = index;
		currentlabel = node_label[index];
		currentstate = node_state[index];

		index+=stride;

		infrontlabel = node_label[index];
		infrontstate = node_state[index];

		///////////////////////////
		//MODIFY BORDER VALUES
		// Set to 0 the label of the border nodes (modifying node_label vector)

		if(loop == 0){  //This operation is performed only once.

			if(node_state[out_index] == 0 && node_state[out_index * params.s] == 0){
				node_label[out_index] = 0;                             //cube side having z=0
				node_label[out_index +  stride * (params.s-1)] = 0 ;   //cube side having z=params.s
			}
			else {
				printf("WARNING: il bordo %d  e' pieno", out_index);
				return;
			}

			//spostiamo il fronte di threads lungo z e verifichiamo quale threads e' di bordo
			// verifico se il threads index punta ad un un elemento di bordo
			// in quel caso se la cella di bordo e' vuota (deve essere vuota) modifico automaticamente il valore del label per quella cella impostandolo a 0

			int use_index = out_index + stride; //move threads front

			if(x == 0 || x == params.s-1 || y == 0 || y == params.s-1){

				for(int i = radius; i<params.s-radius; i++){  //mi sposto lungo z

					if(node_state[use_index] == 0) node_label[use_index] = 0;
					else {
						printf("WARNING: il bordo %d  e' pieno", use_index);
						return;
					}

					use_index +=stride;
				}
			}
			__syncthreads();

		} // end if(loop)

		//////////////////////////////////////////////////////////////////////


		index +=stride;

		for(int i = radius; i<params.s-radius; i++)  //move along z
		{
			/////////////////
			//advance the slice (move the thread front)
			behindlabel = currentlabel;
			currentlabel = infrontlabel;
			infrontlabel = node_label[index];

			behindstate = currentstate;
			currentstate = infrontstate;
			infrontstate = node_state[index];

			index +=stride;
			out_index += stride;

			__syncthreads();

			////////////////
			//update the shared memory tiles
			// fill the halos
			if(threadIdx.y < radius)  //halo above/below
			{
				s_data_label[threadIdx.y][tx]                  = node_label[out_index-radius*params.s];
				s_data_label[threadIdx.y + BDIMY + radius][tx] = node_label[out_index+BDIMY*params.s];

				s_data_state[threadIdx.y][tx]                 = node_state[out_index-radius*params.s];
				s_data_state[threadIdx.y+ BDIMY + radius][tx] = node_state[out_index+BDIMY*params.s];

			}
			if(threadIdx.x < radius) //halo left-right
			{
				s_data_label[ty][threadIdx.x]                 = node_label[out_index-radius];
				s_data_label[ty][threadIdx.x+ BDIMX + radius] = node_label[out_index+BDIMX];

				s_data_state[ty][threadIdx.x]                 = node_state[out_index-radius];
				s_data_state[ty][threadIdx.x+ BDIMX + radius] = node_state[out_index+BDIMX];

			}

			//fill the tale
			s_data_label[ty][tx] = currentlabel;
			s_data_state[ty][tx] = currentstate;

			__syncthreads();

			////////////
			// Calculate the lower label on the list of cells that share the same status
            ////////////
			int oldlabel = currentlabel;

			if((x != 0 || x < params.s-1) && (y != 0 || y < params.s-1)){  // nn eseguo per i threads nel bordo.

				//check the label bottom,top
				// eventualmente inserire l'opzione s_data_label[][] != -1
				if(s_data_state[ty-1][tx] == currentstate && s_data_label[ty-1][tx] < currentlabel)
					currentlabel = s_data_label[ty-1][tx];

				if(s_data_state[ty+1][tx] == currentstate && s_data_label[ty+1][tx] < currentlabel)
					currentlabel = s_data_label[ty+1][tx];

				//check the label right,left
				if(s_data_state[ty][tx+1] == currentstate && s_data_label[ty][tx+1] < currentlabel)
					currentlabel = s_data_label[ty][tx+1];

				if(s_data_state[ty][tx-1] == currentstate && s_data_label[ty][tx-1] < currentlabel)
					currentlabel = s_data_label[ty][tx-1];

				//check the label infront,behind
				if(infrontstate == currentstate && infrontlabel < currentlabel)
					currentlabel = infrontlabel;

				if(behindstate == currentstate && behindlabel < currentlabel)
					currentlabel = behindlabel;

				if(currentlabel != oldlabel) *m_d = true;

				node_label[out_index] = currentlabel;

			}
		} //and for along z axis
	} /* end if on thread index */
}

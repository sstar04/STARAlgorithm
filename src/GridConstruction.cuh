//reduce a pair of bounding boxes (a,b) to a bounding box containing ll and ur
//functor definition
struct bbox_reduction : public thrust::binary_function<bbox, bbox, bbox>
{
	__host__ __device__
	bbox operator()(bbox a, bbox b)
	{
		//lower left corner (punto ha per coordinate il valore minimo per ogni asse)
		point4d ll(thrust::min(a.first.x, b.first.x),thrust::min(a.first.y, b.first.y),thrust::min(a.first.z, b.first.z), 0 );

		//upper right corner (punto ha per coordinate il valore massimo di ogni asse)
		point4d ur(thrust::max(a.second.x, b.second.x),thrust::max(a.second.y, b.second.y),thrust::max(a.second.z, b.second.z), 0);

		return bbox(ll, ur);
	}
};
//convert a point to a bbox containing that point (point) ->(point,point)
//unary functor
struct bbox_transform : public thrust::unary_function<point4d,bbox>
{
	__host__ __device__
	bbox operator()(point4d point)
	{
		return bbox(point, point);
	}

};



//increment function with delta>0
float sum(float a, float delta)
{
	delta = abs(delta);
	if(a<0) return -(abs(a)+delta);
	return a+delta;
}


//Space partitioning

bbox Bbox_EnlargeDivide(bbox a)
{
	//calculate the sides of the box considering that bbox=(min, max);
	point4d D = a.second - a.first;

	//calculate the new points defining the bounding box.
	globalParams.bboxSide =0;
	if(D.x> globalParams.bboxSide) globalParams.bboxSide=D.x;
	if(D.y> globalParams.bboxSide) globalParams.bboxSide=D.y;
	if(D.z> globalParams.bboxSide) globalParams.bboxSide=D.z;  //calculate the maximum side

	int s_temp;
    s_temp = ceil(globalParams.bboxSide/globalParams.cellSide);

    //scale the bboxSide and add an empty border
	globalParams.bboxSide = globalParams.cellSide*s_temp + 2*globalParams.cellSide;

	//set the global parameter
	globalParams.s = s_temp+2;

	//set the number of boxes
    globalParams.NtotCell = globalParams.s*globalParams.s*globalParams.s;

	 point4d max(globalParams.bboxSide, globalParams.bboxSide , globalParams.bboxSide , 0);

    //point4d Increment = D - max; //is the increment that each side needs in order to reach the maximum value.
	point4d Increment = max-D;


	//calculate the new bbox(bmin,bmax)
	point4d bmin(sum(a.first.x,Increment.x/2), sum(a.first.y, Increment.y/2), sum(a.first.z,Increment.z/2), 0);

	point4d bmax(sum(a.second.x,Increment.x/2), sum(a.second.y, Increment.y/2), sum(a.second.z,Increment.z/2), 0);

	//Coordinate that will be used to translate the bbox to the positive semispace
	globalParams.originTranslation.x = bmin.x;
	globalParams.originTranslation.y = bmin.y;
	globalParams.originTranslation.z = bmin.z;

	/*
   //check the size of the box
   point4d D2 = bmin - bmax;
    //print output;
    cout << "box sides " << fixed;
    cout << "(" << D2.x << "," << D2.y << "," << D2.z << ")" << endl ;
*/
	return bbox(bmin, bmax);

}

////////
//CUDA KERNEL
/////////

// calculate position in uniform grid
__device__ int3 calcGridPos(point4d p)
{
	int3 gridPos;

	gridPos.x = floor((p.x - params.originTranslation.x) / params.cellSide);
	gridPos.y = floor((p.y - params.originTranslation.y) / params.cellSide);
	gridPos.z = floor((p.z - params.originTranslation.z) / params.cellSide);
	return gridPos;
}

// calculate address in grid from position (clamping to edges)
__device__ unsigned int calcGridHash(int3 gridPos)
{
	// gridPos.x = gridPos.x & (params.gridSize.x-1);  // wrap grid, assumes size is power of 2
	//  gridPos.y = gridPos.y & (params.gridSize.y-1);
	// gridPos.z = gridPos.z & (params.gridSize.z-1);
	return (gridPos.x * params.s * params.s) + (gridPos.y * params.s) + gridPos.z ;
}


// calculate grid hash value for each particle
__global__
void calcHashD(unsigned int   *gridParticleHash,  // output
		unsigned int   *gridParticleIndex,        // output
		point4d *pos,                             // input: positions
		int   numParticles)
{
	uint index;

	uint x = threadIdx.x + blockIdx.x * blockDim.x ;
	uint y = threadIdx.y + blockIdx.y * blockDim.y ;
	index = x + y * (blockDim.x * gridDim.x);

	if (index >= numParticles) return;

	volatile point4d p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(p);
	unsigned int hash = calcGridHash(gridPos);

	// store grid hash and particle index
	gridParticleHash[index] = hash;
	gridParticleIndex[index] = index;

	return;
}


// calculate grid hash value for each particle
__global__
void calcIDParticle(
		unsigned int   *gridParticleIndex, // output
		point4d *pos,               // input: positions
		int   numParticles)
{
	//unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	uint index;

	uint x = threadIdx.x + blockIdx.x * blockDim.x ;
	uint y = threadIdx.y + blockIdx.y * blockDim.y ;
	index = x + y * (blockDim.x * gridDim.x);

	if (index >= numParticles) return;

	volatile point4d p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(p);
	//cout << "Posizione della cella nella griglia:" << gridPos.x << " " << gridPos.y << " " << gridPos.z << endl;
	unsigned int hash = calcGridHash(gridPos);

	// store grid hash and particle index
	gridParticleIndex[index] = index;

	return;
}


// calculate grid hash value for each particle
__global__
void calcIDParticle_naive(
		unsigned int   *gridParticleIndex, // output
		point4d *pos,               // input: positions
		int   numParticles)
{
	//unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	uint index = blockIdx.x;

	if (index >= numParticles) return;

     point4d p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(p);
	//cout << "Posizione della cella nella griglia:" << gridPos.x << " " << gridPos.y << " " << gridPos.z << endl;
	unsigned int hash = calcGridHash(gridPos);

	// store grid hash and particle index
	gridParticleIndex[index] = index;

	return;
}

//functor that orders the vector points (input data) according to the ordered gridParticleIndex (input data) and
// calculate the cell size, the number of particles for that cell
//unsigned int CreateCellList( ) :  public thrust::unary_function<unsigned int,unsigned int>



// rearrange particle data into sorted order, and find the start of each cell
// in the sorted hash array
__global__
void reorderDataAndFindCellStartD(uint   *cellStart,        // output: cell start index
		uint   *cellEnd,          // output: cell end index
		point4d *sortedPos,        // output: sorted positions

		uint   *gridParticleHash, // input: sorted grid hashes
		uint   *gridParticleIndex,// input: sorted particle indices
		point4d *oldPos,           // input: position array

		uint    numParticles)
{

	//extern
	__shared__ uint sharedHash[65];    // blockSize + 1 elements
	uint index;

	uint x = threadIdx.x + blockIdx.x * blockDim.x ;
	uint y = threadIdx.y + blockIdx.y * blockDim.y ;
	index = x + y * (blockDim.x * gridDim.x);

	uint hash;

	// handle case when no. of particles not multiple of block size
	if (index < numParticles)
	{
		hash = gridParticleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x+1] = hash;

		if (index > 0 && threadIdx.x == 0)
		{
			// first thread in block must load neighbor particle hash
			sharedHash[0] = gridParticleHash[index-1];
		}
	}

	__syncthreads();  //aspetto che tutti i threads hanno scritto la memoria shared

	//raggruppo i punti in funzione del loro indice di cella
	if (index < numParticles)
	{
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x])
		{
			cellStart[hash] = index;

			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
		{
			cellEnd[hash] = index + 1;
		}
		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = gridParticleIndex[index];

		sortedPos[index] = oldPos[sortedIndex]; //recupero il valore senza la macro FETCH che usa la texture memory
		//float4 pos = FETCH(oldPos, sortedIndex);       // macro does either global read or texture fetch
		// float4 vel = FETCH(oldVel, sortedIndex);       // see particles_kernel.cuh
		//sortedPos[index] = pos;
		//sortedVel[index] = vel;
	}
	return;

}


//transfer data
__global__
void transferPointCoord( point4d *sortedPos,        // output: sorted positions
                         float *sortedCoord,
                         float *sortedRadius,
                         uint numParticles)
{
	 uint i;
	 uint index = blockIdx.y*(blockDim.x*gridDim.x) + (blockIdx.x*blockDim.x+threadIdx.x);
	 if (index < numParticles)
	 	{
		 i = index*3;
		 sortedCoord[i] = sortedPos[index].x;
		 sortedCoord[i+1] = sortedPos[index].y;
		 sortedCoord[i+2] = sortedPos[index].z;
		 sortedRadius[index] = sortedPos[index].r;

	 	}

	return;
}

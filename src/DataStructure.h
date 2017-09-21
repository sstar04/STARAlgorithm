/*
 * DataStructure.h
 *
 *  Created on: Nov 28, 2013
 *      Author: sabry
 */

#ifndef DATASTRUCTURE_H_
#define DATASTRUCTURE_H_

#include <thrust/pair.h>

bool checkGrid = false;
bool checkNNSAlgorithm = false;

const int NLABEL = 100;

typedef unsigned int uint;

// struct that will be stored in constant memory holding the simulation parameters
struct SimParams
{
	unsigned int s;  //number of subdivision of the bounding box;
	unsigned int NtotCell;
	float bboxSide; //dimension in mm of the side of the bounding box
	float cellSide;  //dimension in mm
	float3 originTranslation;

	float ext_coeff; //cell elongation factor
/*
	//default constructor
	SimParams(): s(0), NtotCell(0), bboxSide(0), cellSide(0), originTranslation(0), ext_coeff(0) {}
	SimParams(uint a, uint b, float c, float d, float3 e, float f): s(a), NtotCell(b), bboxSide(c), cellSide(d), originTranslation(e), ext_coeff(f) {}

	inline SimParams operator=(const SimParams& b)
	   {

	     return SimParams(b.s, b.NtotCell, b.bboxSide, b.cellSide, b.originTranslation, b.ext_coeff);
	   }

*/
};


struct point3d
{
   float x, y, z;

   //default constructor
   __host__ __device__
   point3d(): x(0),y(0), z(0){}

   __host__ __device__
   point3d(float _x, float _y, float _z): x(_x), y(_y), z(_z){}

   __host__ __device__
   inline point3d operator-(const point3d& b)
   {

     return point3d((x-b.x), (y-b.y), (z-b.z));
   }

    __host__ __device__
   inline point3d operator+(const point3d& b)
   {

     return point3d((x+b.x), (y+b.y), (z+b.z));
   }

};

struct point4d
{
   float x, y, z, r;

   //default constructor
   __host__ __device__
    point4d(): x(0),y(0), z(0), r(0){}

   __host__ __device__
   point4d(float _x, float _y, float _z, float _r): x(_x), y(_y), z(_z), r(_r){}

   //these operator works only for the spatial coordinates.
   __host__ __device__
   inline point4d operator-(const point4d& b)
   {

     return point4d((x-b.x), (y-b.y), (z-b.z), 0);
   }

    __host__ __device__
   inline point4d operator+(const point4d& b)
   {

     return point4d((x+b.x), (y+b.y), (z+b.z),0);
   }


};


__host__ __device__
 inline int3 operator+(const int3& a,const int3& b)
 {
     int x = a.x+b.x;
     int y = a.y+b.y;
     int z = a.z+b.z;

     int3 c = make_int3(x,y,z);

   return c;
 }
struct is_zero
  {
    __host__ __device__
    bool operator()(int x)
    {
      return x == 0;
    }
  };

//bounding box type
typedef thrust::pair<point4d,point4d> bbox;


SimParams globalParams;

__constant__ SimParams params;

__constant__ uint BorderLabel[NLABEL];

//Round a / b to nearest higher integer value
uint iDivUp(uint a, uint b)
{
	return (a % b != 0) ? (a / b + 1) : (a / b);
}

// compute grid and thread block size for a given number of elements
void computeGridSize(uint n, uint blockSize, uint &numBlocks, uint &numThreads)
{
	//numThreads = min(blockSize, n);
	numThreads = blockSize;
	numBlocks = iDivUp(n, numThreads);
}

/*  static __inline__ __host__ __device__ unsigned int min(unsigned int a, unsigned int b)
   {
     return umin(a, b);
   }*/

#endif /* DATASTRUCTURE_H_ */

/*
 * CellSearch.h
 *
 *  Created on: 04/mar/2015
 *      Author: sabry
 */

#ifndef CELLSEARCH_H_
#define CELLSEARCH_H_
#include <vector>

struct Params4Visual
{
	unsigned int s;  //number of subdivision of the bounding box;
	unsigned int NtotCell;

};

typedef unsigned int uint;

void CellSearch(std::vector<uint>& nodes, Params4Visual* P);


#endif /* CELLSEARCH_H_ */

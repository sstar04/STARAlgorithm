#include <iostream>
#include <vector>
#include "src/CellSearch.h"
#include "vtk_visual.h"

using namespace std;

const int MAXVALUE= 10000;

int main() {

	vector<uint> nodes;
    Params4Visual P;

	CellSearch(nodes, &P);

	cout << "Show results" << endl;
	ShowGraph(nodes, &P);


}

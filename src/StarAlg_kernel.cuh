/*
 * StarAlg_kernel.cuh
 *
 *  Created on: 20/set/2017
 *      Author: Sabrina Stella
 */

#ifndef STARALG_KERNEL_CUH_
#define STARALG_KERNEL_CUH_


extern __device__
uint3 CalcGridPosition(uint index);

extern __device__
void FindListAdjacentsOnly(  uint* hash_value,  //input data
		                     uint index,      //thread index
		                      int* n_id);

__global__
void ListBorderLabelSubgraph(uint* node_s, //input data
                             uint* node_label, //input data
                             int s , //input data
                             uint* list_border_label)  //output data
{
    uint x = threadIdx.x + blockIdx.x * blockDim.x ;
    uint y = threadIdx.y + blockIdx.y * blockDim.y ;
    uint index = x + y * (blockDim.x * gridDim.x);

    int ncells = s*s*s;
    int nullvalue = ncells+1;

    if(index < ncells){
       uint3 coord;
       uint temp_hash = index;

       coord = CalcGridPosition(temp_hash);

       if(coord.x % (s-1) == 0 || coord.y % (s-1) == 0 || coord.z % (s-1) == 0){
    	   if(node_s[temp_hash] == 0){
                 //temp_hash e' un nodo di bordo
    		list_border_label[index] = node_label[temp_hash];
    		//printf("valore d hash che e' ambiente %d %d", index, temp_hash);
    	   }


         else list_border_label[index]= nullvalue;
       }
       else list_border_label[index]= nullvalue;

     } /*end if ncells*/

  return;
};

__global__
void Detect_BorderNodes(uint* node_s, //input data
		uint* hash_value, //input data
		uint ncells,
		int* list_border_nodes)  //output data
		{
	//indicizzo linearmente i threads allo stesso modo con cui indicizzo le cellette di suddivisione dello spazio
	uint index;

	uint x = threadIdx.x + blockIdx.x * blockDim.x ;
	uint y = threadIdx.y + blockIdx.y * blockDim.y ;


	//per ogni valore di hash deve corrispondere un threads
	index = x + y * (blockDim.x * gridDim.x);


	if (index < ncells)
	{
		//ricerco i nodi pieni che hanno almeno un vicino environment
		// controllo che il nodo sia pieno
		int temp_status = node_s[index];
		if(temp_status==1){
			int n_id[6] ; //vettore dove annotare l'indice dei vicini ipercubici con lo stesso stato per ogni nodo

			//RICALCOLO TUTTE LE VOLTE LA LISTA DEGLI ADIACENTI USANDO L'HASH PER
			//find the list of adjacent with the same state: fill the vector n_id
			FindListAdjacentsOnly(hash_value, index, n_id);

			bool save = false;
			for(int i= 0 ; i<6 ; i++) {
				//se almeno uno dei nodi adiacenti e` ambiente (=2) allora salva il valore del nodo associato al threads
				if(node_s[n_id[i]] == 2){
					save = true;
				}
				if(n_id[i] == -1) save = true; //se almeno uno degli indici e' -1 allora ho a che fare con una cella confinante con la bounding box e quindi di bordo
			}

			//if (save == true )list_border_nodes[index] = index;
			if (save == true )list_border_nodes[index] = hash_value[index]; // salvo il valore di hash della cella perche' ad ogni cellula dell'aggregato ho associato un valore di hash.
			// il modo con cui nel kernel genero l'indice lineare (index) potrebbe essere diverso da quello con cui genero il valore di hash
			else list_border_nodes[index] = -1;
		}

		else list_border_nodes[index] = -1; // se il nodo e' ambiente o vuoto allora il suo indice nn va memorizzato
	}//end if ncells

	return;
		}

__global__
void SetEnvironmentState(uint* list_bl, //input data: lista dei labels che appartengono al bordo
		int Nlabels,  //input label: numero totale di label unici nel vettore precedente
		uint* node_l,  //input data
		int s,         //input data
		uint* node_s)  //data that will be modified
{


	uint x = threadIdx.x + blockIdx.x * blockDim.x ;
	uint y = threadIdx.y + blockIdx.y * blockDim.y ;
	uint index = x + y * (blockDim.x * gridDim.x);


	int ncells = s*s*s;

	if(index < ncells){

		for(int i = 0; i< Nlabels ; i++ ){
			if(node_l[index] == list_bl[i]) node_s[index] = 2;
		}

	}/*end if ncells*/

	return;
}

__global__
void Find_ListBorderCells ( point4d* sortedPoints, //input data
		uint* cellStart, //input data
		uint* cellStop, //input data : questi ultimi servono per poter navigare nel vettore sortedPoints
		int* list_border_nodes,  //input data: da qui recupero i valori di hash delle cellette di bordo
		bool* border_cell,  //  output data: contene valori veri in  corrispondenza di una cella di bordo,
		uint NCELLS
		//i valori di partenza di border_cell sono tutti nulli
)
{
	//indicizzo linearmente i threads allo stesso modo con cui indicizzo le cellette di suddivisione dello spazio
	uint index;

	uint x = threadIdx.x + blockIdx.x * blockDim.x ;
	uint y = threadIdx.y + blockIdx.y * blockDim.y ;


	//per ogni valore di hash deve corrispondere un threads
	index = x + y * (blockDim.x * gridDim.x);

	if (index < NCELLS)
	{
		int border_hash = list_border_nodes[index] ;
		if(border_hash != -1 ){  //recupero le posizioni delle celle con quell'hash attraverso i vettori cellStart e cellStop

			uint start, stop;
			start = cellStart[border_hash];
			stop = cellStop[border_hash];

			//salvo i dati nel vettore di output
			for(int i = start; i< stop; i ++){
				border_cell[i] = true;  //sto salvando nella memoria global e in modo casuale!
				if(checkk) {
					printf("valore di hash %d = indice di cellule con quell'hash:%d , valore del booleano %B", border_hash, i, border_cell);
					checkk=false;
				}
			}
		}
	} //end if ncells

	return;
}


#endif /* STARALG_KERNEL_CUH_ */

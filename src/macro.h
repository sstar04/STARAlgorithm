#define BLOCK_DIMX 32
#define BLOCK_DIMY 16

#define CUDA_CALL( call) {                                               \
    cudaError err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}

#define CUDA_CHECK_ERROR(errorMessage) {                                    \
    cudaError_t err = cudaGetLastError();                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error: %s in file '%s' in line %i : %s.\n",    \
                errorMessage, __FILE__, __LINE__, cudaGetErrorString( err) );\
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
}


#define GRID_DIMX(n) (((n)+BLOCK_DIMX-1)/BLOCK_DIMX)
#define GRID_DIMY(n) (((n)+BLOCK_DIMY-1)/BLOCK_DIMY)


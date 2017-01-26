// MP 3: Due Sunday, Dec 30, 2012 at 11:59 p.m. PST
#include    <wb.h>

#define wbCheck(stmt) do {                                 \\
        cudaError_t err = stmt;                            \\
        if (err != cudaSuccess) {                          \\
            wbLog(ERROR, \"Failed to run stmt \", #stmt);    \\
            return -1;                                     \\
        }                                                  \\
    } while(0)

const int TILE_WIDTH = 16;
      
// Compute C = A * B
__global__ void matrixMultiplyShared(float * M, float * N, float * P,
			             int numMRows, int numMColumns,
			             int numNRows, int numNColumns,
			             int numPRows, int numPColumns) {
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
      __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
      __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

      int tx = threadIdx.x;
      int ty = threadIdx.y;
  
      int bx = blockIdx.x;
      int by = blockIdx.y;
  
      int Row = by * TILE_WIDTH + ty;
      int Col = bx * TILE_WIDTH + tx;  
 
      float Pvalue = 0;
  
      for (int m = 0; m < (numMColumns-1)/TILE_WIDTH+1; ++m) {

        if (Row < numMRows && m*TILE_WIDTH+tx < numMColumns) { // war tx
          Mds[ty][tx] = M[Row*numMColumns + m*TILE_WIDTH +tx]; // war tx
        } else {
          Mds[ty][tx] = 0; 
        }

        if (m*TILE_WIDTH+ty < numNRows && Col < numNColumns) { // war ty
          Nds[ty][tx] = N[(m*TILE_WIDTH+ty) * numNColumns+Col];   // war ty
        } else {
          Nds[ty][tx] = 0;
        }

        __syncthreads();

        if (Row < numPRows && Col < numPColumns) {
          for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
          }
        }

      __syncthreads();

      }
      
  
      if (Row < numPRows && Col < numPColumns) {
        P[Row*numPColumns + Col] = Pvalue;
      }
  
}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, \"Importing data and creating memory on host\");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;
    numCColumns = numBColumns;
    //@@ Allocate the hostC matrix
    int sizeA = numARows * numAColumns * sizeof(float);
    int sizeB = numBRows * numBColumns * sizeof(float);
    int sizeC = numCRows * numCColumns * sizeof(float);
  
    hostC = (float *) malloc(sizeC);
      
    wbTime_stop(Generic, \"Importing data and creating memory on host\");

    wbLog(TRACE, \"The dimensions of A are \", numARows, \" x \", numAColumns);
    wbLog(TRACE, \"The dimensions of B are \", numBRows, \" x \", numBColumns);

    wbTime_start(GPU, \"Allocating GPU memory.\");
    //@@ Allocate GPU memory here
    wbCheck(cudaMalloc((void**) &deviceA, sizeA));
    wbCheck(cudaMalloc((void**) &deviceB, sizeB));
    wbCheck(cudaMalloc((void**) &deviceC, sizeC));

    wbTime_stop(GPU, \"Allocating GPU memory.\");

    wbTime_start(GPU, \"Copying input memory to the GPU.\");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice));
  
    wbTime_stop(GPU, \"Copying input memory to the GPU.\");
    
    //@@ Initialize the grid and block dimensions here
    int BLOCK_SIZE = 16;
    dim3 dimGrid((numCColumns + BLOCK_SIZE - 1)/BLOCK_SIZE, (numCRows + BLOCK_SIZE - 1)/BLOCK_SIZE, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
  
    wbTime_start(Compute, \"Performing CUDA computation\");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared<<<dimGrid,dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns); 
  
    cudaThreadSynchronize();
    wbTime_stop(Compute, \"Performing CUDA computation\");
    
    wbTime_start(Copy, \"Copying output memory to the CPU\");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaGetLastError());
    wbCheck(cudaMemcpy(hostC, deviceC, sizeC, cudaMemcpyDeviceToHost));
  
    wbTime_stop(Copy, \"Copying output memory to the CPU\");

    wbTime_start(GPU, \"Freeing GPU Memory\");
    //@@ Free the GPU memory here
    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
  
    wbTime_stop(GPU, \"Freeing GPU Memory\");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

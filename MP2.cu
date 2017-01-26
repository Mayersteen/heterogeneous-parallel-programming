// MP 2: Due Sunday, Dec 16, 2012 at 11:59 p.m. PST
#include    <wb.h>

#define wbCheck(stmt) do {                                 \\
        cudaError_t err = stmt;                            \\
        if (err != cudaSuccess) {                          \\
            wbLog(ERROR, \"Failed to run stmt \", #stmt);    \\
            return -1;                                     \\
        }                                                  \\
    } while(0)

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
			       int numARows, int numAColumns,
			       int numBRows, int numBColumns,
			       int numCRows, int numCColumns) {
  
  //@@ Insert code to implement matrix multiplication here
    
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < numCRows && col < numCColumns) {
  
    float temp = 0.0;
    
    for (int k = 0; k < numAColumns; ++k) {
      temp += A[row * numAColumns + k] * B[k * numBColumns + col];
      
    }
    
    C[row * numBColumns + col]= temp;
    
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
//    wbCheck(cudaMemset(deviceC, 0, sizeC));
      
    wbTime_stop(GPU, \"Allocating GPU memory.\");

    wbTime_start(GPU, \"Copying input memory to the GPU.\");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy(deviceA, hostA, sizeA, cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, sizeB, cudaMemcpyHostToDevice));
      
    wbTime_stop(GPU, \"Copying input memory to the GPU.\");
    
    //@@ Initialize the grid and block dimensions here
    int BLOCK_SIZE = 16;
  
//    dim3 dimGrid(ceil((float)(numBColumns + TILE_WIDTH - 1)/TILE_WIDTH), ceil((float)(numARows + TILE_WIDTH - 1)/TILE_WIDTH), 1);
//    dim3 dimGrid(ceil((float)(numAColumns)/TILE_WIDTH), ceil((float)(numBRows)/TILE_WIDTH), 1);
//    dim3 dimGrid(ceil(numARows/(float)BLOCK_SIZE), ceil(numBColumns/(float)BLOCK_SIZE), 1);
//    dim3 dimGrid(512,512, 1);
//    dim3 dimGrid(numARows/BLOCK_SIZE, numBColumns/BLOCK_SIZE, 1);
//+    dim3 dimGrid(ceil(numAColumns / (float)BLOCK_SIZE), ceil(numBRows / (float)BLOCK_SIZE), 1);
  
    dim3 dimGrid((numCRows + BLOCK_SIZE - 1)/BLOCK_SIZE, (numCColumns + BLOCK_SIZE - 1)/BLOCK_SIZE, 1);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1);
    
//    wbLog(TRACE, \"The dimensions of C are \", numCRows, \" x \", numCColumns);
//    wbLog(TRACE, \"dimGrid(\", ceil(numCRows/BLOCK_SIZE),\",\",ceil(numCColumns/BLOCK_SIZE),\",1)\");
  
    wbTime_start(Compute, \"Performing CUDA computation\");
    //@@ Launch the GPU Kernel here
    matrixMultiply<<<dimGrid,dimBlock>>>(deviceA, deviceB, deviceC, numARows, numAColumns, numBRows, numBColumns, numCRows, numCColumns); 
                      
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
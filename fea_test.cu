#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>


#define M 16  //size of matrix A M by N
#define N 16
#define NE 18 //number of elements

#define BLOCK_X 6 // number of integration points
#define BLOCK_Y 9 // number of expressions
#define BLOCK_Z ((int)(32*32)/(6*9)) //number of elements in a block
#define NDOF 3 //number of DOFs
#define NNODE 3 //number of nodes

//TODO
__constant__ float triW[7] = { 0.06296959f, 0.06619708f, 0.06296959f, 0.06619708f, 0.06296959f, 0.06619708f, 0.11250000f };
__constant__ float triR[7] = { 0.10128651f, 0.47014206f, 0.79742699f, 0.47014206f, 0.10128651f, 0.05971587f, 0.33333333f };
__constant__ float triS[7] = { 0.10128651f, 0.05971587f, 0.10128651f, 0.47014206f, 0.79742699f, 0.47014206f, 0.33333333f };
__constant__ float triT[7] = { 
	1.0f-triR[0]-triS[0],
	1.0f-triR[1]-triS[1],
	1.0f-triR[2]-triS[2],
	1.0f-triR[3]-triS[3],
	1.0f-triR[4]-triS[4],
	1.0f-triR[5]-triS[5],
	1.0f-triR[6]-triS[6]
};


//This function should be generated from the symbol expressions of the integrand
__device__ float integrand(int funIdx, float *params)
{
	float x1 = params[0];
	float x2 = params[1];
	float x3 = params[2];
	float y1 = params[3];
	float y2 = params[4];
	float y3 = params[5];
	float r = params[6];
	float s = params[7];
	float t = params[8];
	//x = x1*r + x2*s + x3*t
	//y = y1*r + y2*s + y3*t
	float xr = x1-x3;
	float xs = x2-x3;
	float yr = y1-y3;
	float ys = y2-y3;
	// jac = (xr xs)
	//       (yr ys)
	float jac = xr*ys-xs*yr;
	
	float rx =  ys/jac;
	float ry = -xs/jac;
	float sx = -yr/jac;
	float sy =  xr/jac;
	float tx = -rx - sx;
	float ty = -ry - sy;

//	float[][] lhs = {
//			{rx*rx + ry*ry, rx*sx + ry*sy, rx*tx + ry*ty},
//			{sx*rx + sy*ry, sx*sx + sy*sy, sx*tx + sy*ty},
//			{tx*rx + ty*ry, tx*sx + ty*sy, tx*tx + ty*ty},
//	};
	if(funIdx == 0)
		return (rx*rx + ry*ry)*jac*0.5;
	if(funIdx == 1)
		return (rx*sx + ry*sy)*jac*0.5;
	if(funIdx == 2)
		return (rx*tx + ry*ty)*jac*0.5;
	if(funIdx == 3)
		return (sx*rx + sy*ry)*jac*0.5;
	if(funIdx == 4)
		return (sx*sx + sy*sy)*jac*0.5;
	if(funIdx == 5)
		return (sx*tx + sy*ty)*jac*0.5;
	if(funIdx == 6)
		return (tx*rx + ty*ry)*jac*0.5;
	if(funIdx == 7)
		return (tx*sx + ty*sy)*jac*0.5;
	if(funIdx == 8)
		return (tx*tx + ty*ty)*jac*0.5;
	return 0.0f;
}

//Version 2: user global memory directly
__global__ void fea_kernel(double* A, 
		double *X, double *Y, // (x,y) of each element for all the element
		int *gIdx // node index of each element for all the element
	)
{
	int eleIdx = blockIdx.x * BLOCK_Z + threadIdx.z; //global element index 
	//local matrix row and column index
	//threadIdx.y = 0,1,2,3,4,5,6,7,8
	int li = threadIdx.y / NDOF;
	int lj = threadIdx.y / NDOF;
	__shared__ localFlatMatrix[BLOCK_Y*BLOCK_Z]; //array for the local flat matrices of all the elememnts in the current block
	int lfmIdx = threadIdx.z*BLOCK_Y + threadIdx.y; //local flat matrix index of the integrand of threadIdx.y
	float params[3*NNODE]; //parameters array of integrand

	//compute local matrix
	if(eleIdx < NE)
	{
		params[0] = X[NNODE*eleIdx+0];
		params[1] = X[NNODE*eleIdx+1];
		params[2] = X[NNODE*eleIdx+2];
		params[3] = Y[NNODE*eleIdx+0];
		params[4] = Y[NNODE*eleIdx+1];
		params[5] = Y[NNODE*eleIdx+2];
		params[6] = triR[threadIdx.x];
		params[7] = triS[threadIdx.x];
		params[8] = triT[threadIdx.x]; //triT[threadIdx.x]=1.0-triR[threadIdx.x]-triS[threadIdx.x];
		atomicAdd( &localFlatMatrix[lfmIdx], triW[threadIdx.x]*integrand(threadIdx.x, params) );
	}
	__syncthreads();

	//write to gobal matrix A
	if(eleIdx < NE)
	{
		if(threadIdx.x == 0)
		{
			//global matrix row and col index
			int gi  = gIdx[NNODE*eleIdx + li];
			int gj  = gIdx[NNODE*eleIdx + lj];
			atomicAdd( &A[ N*gj + gi ], localFlatMatrix[lfmIdx] );
		}
	}
}

cudaError_t assembleWithCuda()
{
    dim3 dim_block;
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    float elapsed = 0;

    dim_block.x = BLOCK_X;
    dim_block.y = BLOCK_Y;
    dim_block.z = BLOCK_Z;

    printf("block_x:%d, block_y:%d, block_z:%d\n", dim_block.x, dim_block.y, dim_block.z);

    cudaSetDevice(0);

    float *A  = (float*)malloc( M*N*sizeof(float) );
    float *X  = (float*)malloc( NE*NNODE*sizeof(float) );
    float *Y  = (float*)malloc( NE*NNODE*sizeof(float) );
    int *gIdx = (int*)malloc( NE*NNODE*sizeof(int) );

    float *dA = NULL;
    cudaMalloc((void**)&dA, M*N*sizeof(float));
    float *dX = NULL;
    cudaMalloc((void**)&dX, NE*NNODE*sizeof(float));
    float *dY = NULL;
    cudaMalloc((void**)&dY, NE*NNODE*sizeof(float));
    int *dGIdx = NULL;
    cudaMalloc((void**)&dGIdx, NE*NNODE*sizeof(int));

    cudaMemcpy(dA, A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dX, X, NE*NNODE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, NE*NNODE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dGIdx, gIdx, NE*NNODE*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fea_kernel << <1, dim_block >> >(dA, dX, dY, dGIdx);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    printf("GPU Time: %f ms\n", elapsed);

    cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(A, dA, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dGIdx);

    return cudaStatus;
}


int main()
{
    assembleWithCuda();
    cudaDeviceReset();
    return 0;
}

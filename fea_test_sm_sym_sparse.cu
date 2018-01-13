//share memory + hard code symbol output
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <vector>

using namespace std;

#define MESH_W 500
#define MESH_H 500

#define M (MESH_W+1)*(MESH_H+1) //size of matrix A M by N
#define N (MESH_W+1)*(MESH_H+1)
#define NE 2*MESH_W*MESH_H //number of elements

#define K20 192
#define BLOCK_X 7 // number of integration points
#define BLOCK_Y 9 // number of expressions
//#define BLOCK_Z ((int)(32*32)/(BLOCK_X*BLOCK_Y)) //number of elements in a block
#define BLOCK_Z (K20/(BLOCK_X*BLOCK_Y)) //number of elements in a block
#define NDOF 3 //number of DOFs
#define NNODE 3 //number of nodes

__constant__ float triW[7] = { 0.06296959f, 0.06619708f, 0.06296959f, 0.06619708f, 0.06296959f, 0.06619708f, 0.11250000f };
__constant__ float triR[7] = { 0.10128651f, 0.47014206f, 0.79742699f, 0.47014206f, 0.10128651f, 0.05971587f, 0.33333333f };
__constant__ float triS[7] = { 0.10128651f, 0.05971587f, 0.10128651f, 0.47014206f, 0.79742699f, 0.47014206f, 0.33333333f };
__constant__ float triT[7] = { 0.79742698f, 0.47014207f, 0.1012865f,  0.05971588f, 0.1012865f,  0.47014207f, 0.33333334f };

class Node 
{
public:
  double x, y, z;
  int flag; //boundary flag
  int index; //global index of node
};

class Element 
{
public:
  vector<Node*> nodes;
};

class Mesh 
{
public:
  vector<Node*> nodes;
  vector<Element*> elements;
  void printMesh()
  {
    cout << "number of nodes = " << nodes.size() << endl;
    for(int i=0; i<nodes.size(); i++)
    {
      Node *node = nodes[i];
      cout << node->index << " " <<node->x << " " << node->y << " " << node->flag << endl;
    }
    cout << "number of elements = " <<elements.size() << endl;
    for(int i=0; i<elements.size(); i++)
    {
      Element *e = elements[i];
      cout << e->nodes[0]->index << " " << e->nodes[1]->index << " " << e->nodes[2]->index << endl;
    }
  }
};

class RectangleMesh : public Mesh 
{
public:
  double x0,x1,y0,y1;
  int nRow, nCol;
  RectangleMesh(double x0, double x1, double y0, double y1, int nRow, int nCol) 
  {
    this->x0 = x0;
    this->x1 = x1;
    this->y0 = y0;
    this->y1 = y1;
    this->nRow = nRow;
    this->nCol = nCol;
    generate();
  }

  void generate()
  {
    double stepx = (x1-x0)/nCol;
    double stepy = (y1-y0)/nRow;
    //generate nodes
    for(int i=0; i<=nRow; i++)
    {
      double y = y0+i*stepy;
      for(int j=0; j<=nCol; j++)
      {
        double x = x0+j*stepx;
        Node *node = new Node();
        node->x = x;
        node->y = y;
        if(i==0 || i==nRow || j==0 || j==nCol)
          node->flag = 1; //on the bounday
        else
          node->flag = 0;
        node->index = i*(nCol+1) + j;
        nodes.push_back(node);
      }
    }
    //generate elements
    for(int i=0; i<nRow; i++)
    {
      for(int j=0; j<nCol; j++)
      {
        Element *e = new Element();
        int n1 = i*(nCol+1) + j;
        int n2 = n1 + 1;
        int n3 = (i+1)*(nCol+1) + j;
        e->nodes.push_back(nodes[n1]);
        e->nodes.push_back(nodes[n2]);
        e->nodes.push_back(nodes[n3]);
        elements.push_back(e);

        e = new Element();
        n1 = i*(nCol+1) + j + 1;
        n2 = (i+1)*(nCol+1) + j+ 1;
        n3 = n2 - 1;
        e->nodes.push_back(nodes[n1]);
        e->nodes.push_back(nodes[n2]);
        e->nodes.push_back(nodes[n3]);
        elements.push_back(e);
      }
    }
  }
};

class UnitSquareMesh : public RectangleMesh
{
public:
  UnitSquareMesh(int nRow, int nCol) :
    RectangleMesh(0.0,1.0,0.0,1.0,nRow,nCol) {}
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
	//printf("%d %f %f %f %f %f %f %f %f %f\n", funIdx, x1,x2,x3,y1,y2,y3,r,s,t);
	if(funIdx == 0)
		return -( 1.0/powf( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3),2.0)*powf( x2-x3,2.0)+powf( y2-y3,2.0)/powf( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3),2.0))*( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3));
	if(funIdx == 1)
		return ( ( x1-x3)/powf( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3),2.0)*( x2-x3)+( y1-y3)*( y2-y3)/powf( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3),2.0))*( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3));
	if(funIdx == 2)
		return ( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))*( ( y2-y3)*( ( y2-y3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-( y1-y3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3)))/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-( ( x1-x3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-1.0/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))*( x2-x3))/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))*( x2-x3));
	if(funIdx == 3)
		return ( ( x1-x3)/powf( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3),2.0)*( x2-x3)+( y1-y3)*( y2-y3)/powf( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3),2.0))*( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3));
	if(funIdx == 4)
		return -( powf( x1-x3,2.0)/powf( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3),2.0)+powf( y1-y3,2.0)/powf( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3),2.0))*( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3));
	if(funIdx == 5)
		return ( ( ( x1-x3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-1.0/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))*( x2-x3))*( x1-x3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-( y1-y3)*( ( y2-y3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-( y1-y3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3)))/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3)))*( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3));
	if(funIdx == 6)
		return ( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))*( ( y2-y3)*( ( y2-y3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-( y1-y3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3)))/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-( ( x1-x3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-1.0/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))*( x2-x3))/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))*( x2-x3));
	if(funIdx == 7)
		return ( ( ( x1-x3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-1.0/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))*( x2-x3))*( x1-x3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-( y1-y3)*( ( y2-y3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-( y1-y3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3)))/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3)))*( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3));
	if(funIdx == 8)
		return -( powf( ( x1-x3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-1.0/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))*( x2-x3),2.0)+powf( ( y2-y3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3))-( y1-y3)/( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3)),2.0))*( ( y1-y3)*( x2-x3)-( y2-y3)*( x1-x3));
	return 0.0f;
}

__global__ void fea_kernel2(float* A,
    int *rowA, int *colA,
                float *X, float *Y, // (x,y) of each element for all the element
                int *gIdx // node index of each element for all the element
        )
{
  printf("%d %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x);
}

//Version 3: use shared memory
__global__ void fea_kernel(float* A,
    int *rowA, int *colA,
		float *X, float *Y, // (x,y) of each element for all the element
		int *gIdx // node index of each element for all the element
	)
{
	__shared__ float sX[BLOCK_Z*NNODE];    //shared memory of copy of X in the current block
	__shared__ float sY[BLOCK_Z*NNODE];    //shared memory of copy of Y in the current block
	__shared__ float sGIdx[BLOCK_Z*NNODE]; //shared memory of copy of gIdx in the current block

	int gEleIdx = BLOCK_Z*blockIdx.x + threadIdx.z; //global element index 
	int sEleIdx = NNODE*threadIdx.z;                //global element index in shared memory arrays: sX,sY,sGIdx

	// copy from global memory to shared memory for X, Y and gIdx
	if(threadIdx.x==0 && threadIdx.y==0)
	{
#pragma unroll
		for(int i=0; i<NNODE; i++)
			sX[sEleIdx+i]=X[NNODE*gEleIdx+i];

#pragma unroll
		for(int i=0; i<NNODE; i++)
			sY[sEleIdx+i]=Y[NNODE*gEleIdx+i];

#pragma unroll
		for(int i=0; i<NNODE; i++)
			sGIdx[sEleIdx+i]=gIdx[NNODE*gEleIdx+i];
	}
	__syncthreads();

	//local matrix row and column index
	//threadIdx.y = 0,1,2,3,4,5,6,7,8 (BLOCK_Y)
	int li = threadIdx.y / NDOF;
	int lj = threadIdx.y % NDOF;
	__shared__ float localFlatMatrix[BLOCK_Y*BLOCK_Z]; //array for the local flat matrices of all the elememnts in the current block
	int lfmIdx = threadIdx.z*BLOCK_Y + threadIdx.y; //local flat matrix index of the integrand of threadIdx.y
	float params[3*NNODE]; //parameters array of integrand

	//compute local matrix
	if(gEleIdx < NE)
	{
#pragma unroll
		for(int i=0; i<NNODE; i++)
			params[i] = sX[sEleIdx+i];

#pragma unroll
		for(int i=0; i<NNODE; i++)
			params[NNODE+i] = sY[sEleIdx+i];

		params[2*NNODE+0] = triR[threadIdx.x];
		params[2*NNODE+1] = triS[threadIdx.x];
		params[2*NNODE+2] = triT[threadIdx.x]; //triT[threadIdx.x]=1.0-triR[threadIdx.x]-triS[threadIdx.x];

		atomicAdd( &localFlatMatrix[lfmIdx], triW[threadIdx.x]*integrand(threadIdx.y, params) );
	}
	__syncthreads();

	//write to gobal matrix A
	if(gEleIdx < NE)
	{
		if(threadIdx.x == 0)
		{
      //global matrix row and column index
      int gi  = sGIdx[sEleIdx + li];
      int gj  = sGIdx[sEleIdx + lj];
      rowA[gEleIdx*(NNODE*NNODE) + threadIdx.y] = gi;
      colA[gEleIdx*(NNODE*NNODE) + threadIdx.y] = gj;
      A[gEleIdx*(NNODE*NNODE) + threadIdx.y]    = localFlatMatrix[lfmIdx];
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

    RectangleMesh mesh(-3.0, 3.0, -3.0, 3.0, MESH_W, MESH_H);
    //mesh.printMesh();

    float *X  = (float*)malloc( NE*NNODE*sizeof(float) );
    float *Y  = (float*)malloc( NE*NNODE*sizeof(float) );
    int *gIdx = (int*)malloc( NE*NNODE*sizeof(int) );
    float *A  = (float*)malloc( NE*(NNODE*NNODE)*sizeof(float) );
    int *rowA = (int*)malloc( NE*(NNODE*NNODE)*sizeof(int) );
    int *colA = (int*)malloc( NE*(NNODE*NNODE)*sizeof(int) );

    for(int i=0; i<mesh.elements.size(); i++)
    {
      Element *e = mesh.elements[i];
      X[NNODE*i+0] = e->nodes[0]->x;
      X[NNODE*i+1] = e->nodes[1]->x;
      X[NNODE*i+2] = e->nodes[2]->x;
      Y[NNODE*i+0] = e->nodes[0]->y;
      Y[NNODE*i+1] = e->nodes[1]->y;
      Y[NNODE*i+2] = e->nodes[2]->y;
      gIdx[NNODE*i+0] = e->nodes[0]->index;
      gIdx[NNODE*i+1] = e->nodes[1]->index;
      gIdx[NNODE*i+2] = e->nodes[2]->index;
    }

    float *dA = NULL;
    cudaMalloc((void**)&dA, NE*(NNODE*NNODE)*sizeof(float));
    int *dRowA = NULL;
    cudaMalloc((void**)&dRowA, NE*(NNODE*NNODE)*sizeof(int));
    int *dColA = NULL;
    cudaMalloc((void**)&dColA, NE*(NNODE*NNODE)*sizeof(int));
    float *dX = NULL;
    cudaMalloc((void**)&dX, NE*NNODE*sizeof(float));
    float *dY = NULL;
    cudaMalloc((void**)&dY, NE*NNODE*sizeof(float));
    int *dGIdx = NULL;
    cudaMalloc((void**)&dGIdx, NE*NNODE*sizeof(int));

    cudaMemcpy(dX, X, NE*NNODE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dY, Y, NE*NNODE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dGIdx, gIdx, NE*NNODE*sizeof(int), cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    fea_kernel << <(NE+BLOCK_Z-1)/BLOCK_Z, dim_block >> >(dA, dRowA, dColA, dX, dY, dGIdx);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);
    printf("GPU Time: %f ms\n", elapsed);

    cudaDeviceSynchronize();

    cudaStatus = cudaMemcpy(A, dA, NE*(NNODE*NNODE)*sizeof(float), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(rowA, dRowA, NE*(NNODE*NNODE)*sizeof(int), cudaMemcpyDeviceToHost);
    cudaStatus = cudaMemcpy(colA, dColA, NE*(NNODE*NNODE)*sizeof(int), cudaMemcpyDeviceToHost);
    for(size_t i=0; i<16; i++) {
        std::cout << "(" << rowA[i] << "," << colA[i] << ")" << " " << A[i] << std::endl;
    }

    cudaFree(dA);
    cudaFree(dRowA);
    cudaFree(dColA);
    cudaFree(dX);
    cudaFree(dY);
    cudaFree(dGIdx);

    return cudaStatus;
}

// nvcc --std=c++11 fea_test_sm_sym_sparse.cu -o fea_test_sm_sym_sparse
// /usr/local/cuda-8.0/bin/nvcc --std=c++11 fea_test_sm_sym_sparse.cu -o fea_test_sm_sym_sparse
// /usr/local/cuda-9.1/bin/nvcc --std=c++11 fea_test_sm_sym_sparse.cu -o fea_test_sm_sym_sparse
int main()
{
    assembleWithCuda();
    cudaDeviceReset();
    return 0;
}

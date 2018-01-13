//global memory only
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <chrono>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <ctime>
#include <vector>

using namespace std;

#define MESH_W 4
#define MESH_H 4

#define M (MESH_W+1)*(MESH_H+1) //size of matrix A M by N
#define N (MESH_W+1)*(MESH_H+1)
#define NE 2*MESH_W*MESH_H //number of elements

#define BLOCK_X 7 // number of integration points
#define BLOCK_Y 9 // number of expressions
#define BLOCK_Z ((int)(32*32)/(BLOCK_X*BLOCK_Y)) //number of elements in a block
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
__global__ void fea_kernel(float* A, 
		float *X, float *Y, // (x,y) of each element for all the element
		int *gIdx // node index of each element for all the element
	)
{
	//if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0)
	//{
	//	for(int i=0; i<NE*NNODE; i++) {
	//		printf("%d (%f %f)\n", gIdx[i], X[i], Y[i]);
	//	}
	//}
	int eleIdx = blockIdx.x * BLOCK_Z + threadIdx.z; //global element index 
	//local matrix row and column index
	//threadIdx.y = 0,1,2,3,4,5,6,7,8
	int li = threadIdx.y / NDOF;
	int lj = threadIdx.y % NDOF; //bugfix:  / => %
	__shared__ float localFlatMatrix[BLOCK_Y*BLOCK_Z]; //array for the local flat matrices of all the elememnts in the current block
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
		//printf("%f ", triW[threadIdx.x]*integrand(threadIdx.y, params));
		atomicAdd( &localFlatMatrix[lfmIdx], triW[threadIdx.x]*integrand(threadIdx.y, params) ); //bugfix: threadIdx.x => threadIdx.y
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
			printf("%d %d %d %f\n", threadIdx.y, gi, gj, localFlatMatrix[lfmIdx]);
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

	RectangleMesh mesh(-3.0, 3.0, -3.0, 3.0, MESH_W, MESH_H);
	mesh.printMesh();

    float *A  = (float*)malloc( M*N*sizeof(float) );
    float *X  = (float*)malloc( NE*NNODE*sizeof(float) );
    float *Y  = (float*)malloc( NE*NNODE*sizeof(float) );
    int *gIdx = (int*)malloc( NE*NNODE*sizeof(int) );

    for(int i=0; i<M*N; i++)
    	A[i] = 0.0f;
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
    fea_kernel << <2, dim_block >> >(dA, dX, dY, dGIdx); //bugfix 1 => 2

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsed, start, stop);

    printf("GPU Time: %f ms\n", elapsed);

    cudaDeviceSynchronize();
    cudaStatus = cudaMemcpy(A, dA, M*N*sizeof(float), cudaMemcpyDeviceToHost);
    for(int i=0; i<M; i++) {
    	for(int j=0; j<N; j++) {
	    	printf("%f ", A[i*N+j]);
    	}
    	printf("\n");
    }

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

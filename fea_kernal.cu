//Version 1: Use individual arrays for each node of the elements (not recommanded)
__global__ void fea_kernel(int M, int N, double* A, 
	int NE, //number of elements
	double *X1, double *Y1, int *N1, // node 1s of each element
	double *X2, double *Y2, int *N2, // node 2s of each element
	double *X3, double *Y3, int *N3 )// ndoe 3s of each element
{
	int gi, gj; //global matrix row and col index

	int paramsStart;
	__shared__ localFlatMatrix[BLOCK_Y * BLOCK_Z]; //

	int eleIdx = blockIdx.x * BLOCK_Z + threadIdx.z; //global element index 

	if(eleIdx < NE)
	{
		if(threadIdx.y == 0)
		{
			//u1 * v1
			params[0] = X1[eleIdx];
			params[1] = X2[eleIdx];
			params[2] = X3[eleIdx];
			params[3] = Y1[eleIdx];
			params[4] = Y2[eleIdx];
			params[5] = Y3[eleIdx];
			paramsStart = 6;
			params[paramsStart]   = triR[i]; 
			params[paramsStart+1] = triS[i]; 
			params[paramsStart+2] = 1.0-triR[i]-triS[i];
			atomicAdd(&localFlatMatrix[threadIdx.z*BLOCK_Y + threadIdx.y], triW[i]*integrand(params) );
		}
		else if(threadIdx.y == 1)
		{
			//u1 * v2
			...
			atomicAdd(&localFlatMatrix[threadIdx.z*BLOCK_Y + threadIdx.y], triW[i]*integrand(params) );
		}
		else if(threadIdx.y == 2)
		{
			//u1 * v3
		}
		else if(threadIdx.y == 3)
		{
			//u2 * v1
		}
		else if(threadIdx.y == 4)
		{
			//u2 * v2
		}
		else if(threadIdx.y == 5)
		{
			//u2 * v3
		}
		else if(threadIdx.y == 6)
		{
			//u3 * v1
		}
		else if(threadIdx.y == 7)
		{
			//u3 * v2
		}
		else if(threadIdx.y == 8)
		{
			//u3 * v3
		}
		
		__syncthreads();

		//write to gobal matrix A
		if(eleIdx < NE)
		{
			if(threadIdx.y == 0)
			{
				//u1 * v1
				gi  = N1[eleIdx];
				gj  = N1[eleIdx];
				if(threadIdx.x == 0)
				{
					atomicAdd(&A[N*gj+gi], localMatrix[threadIdx.z*BLOCK_Y + threadIdx.y] );
				}
			}
			else if(threadIdx.y == 1)
			{
				//u1 * v2
				gi  = N1[eleIdx];
				gj  = N2[eleIdx];
				if(threadIdx.x == 0)
				{
					atomicAdd(&A[N*gj+gi], localMatrix[threadIdx.z*BLOCK_Y + threadIdx.y] );
				}
			}
			else if(threadIdx.y == 2)
			{
				//u1 * v3
			}
			else if(threadIdx.y == 3)
			{
				//u2 * v1
			}
			else if(threadIdx.y == 4)
			{
				//u2 * v2
			}
			else if(threadIdx.y == 5)
			{
				//u2 * v3
			}
			else if(threadIdx.y == 6)
			{
				//u3 * v1
			}
			else if(threadIdx.y == 7)
			{
				//u3 * v2
			}
			else if(threadIdx.y == 8)
			{
				//u3 * v3
			}
		}
	}
}


#define BLOCK_X 6 // number of integration points
#define BLOCK_Y 9 // number of expressions
#define BLOCK_Z ((int)(32*32)/(6*9)) //number of elements in a block
#define NDOF 3 //number of DOFs
#define NNODE 3 //number of nodes

//TODO
__constant__ double triW[7]; // = { 0.06296959, 0.06619708, 0.06296959, 0.06619708, 0.06296959, 0.06619708, 0.11250000 };
__constant__ double triR[7]; // = { 0.10128651, 0.47014206, 0.79742699, 0.47014206, 0.10128651, 0.05971587, 0.33333333 };
__constant__ double triS[7]; // = { 0.10128651, 0.05971587, 0.10128651, 0.47014206, 0.79742699, 0.47014206, 0.33333333 };
__constant__ double triT[7]; // = t=1-r-s


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

/* TODO right hand side
			double[][] intPnW = {
					{0.5, 0.5, 1.0/6.0},
					{0.0, 0.5, 1.0/6.0},
					{0.5, 0.0, 1.0/6.0}
			};
			for(int k=0; k<intPnW.length; k++) {
				double r=intPnW[k][0];
				double s=intPnW[k][1];
				double t=1-r-s;
				double w=intPnW[k][2];
				double x = x1*r+x2*s+x3*t;
				double y = y1*r+y2*s+y3*t;
				//f=-2*(x*x+y*y)+36
				double f = -2.0*(x*x+y*y)+36.0;
				vecb[e.nodes.get(0).getIndex() - 1] += f*r*jac*w;
				vecb[e.nodes.get(1).getIndex() - 1] += f*s*jac*w;
				vecb[e.nodes.get(2).getIndex() - 1] += f*t*jac*w;
			}

		}
*/



//Version 2: user global memory directly
__global__ void fea_kernel(int M, int N, float* A, 
		int NE, //number of elements
		float *X, float *Y, // (x,y) of each element for all the element
		int *gIdx // node index of each element for all the element
	)
{
	int eleIdx = blockIdx.x * BLOCK_Z + threadIdx.z; //global element index 
	//local matrix row and column index
	//threadIdx.y = 0,1,2,3,4,5,6,7,8
	int li = threadIdx.y / NDOF;
	int lj = threadIdx.y % NDOF;
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
		atomicAdd( &localFlatMatrix[lfmIdx], triW[threadIdx.x]*integrand(threadIdx.y, params) );
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
			atomicAdd( &A[N*gj + gi], localFlatMatrix[lfmIdx] );
		}
	}
}


//Version 3: use shared memory
__global__ void fea_kernel(int M, int N, float* A, 
		int NE, //number of elements
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
			atomicAdd( &A[N*gj + gi], localFlatMatrix[lfmIdx] );
		}
	}
}





//ginac + template + sparse matrix + atomicAdd
#include <iostream>
#include <time.h>
#include <vector>
#include <ginac/ginac.h>
#include <chrono>
#include <sstream> // std::ostringstream
#include <stdio.h>
#include <nvrtc.h>
#include <cuda.h>
#include <string>
#include <set>
#include <algorithm>    // std::sort

using namespace GiNaC;
using namespace std;

#define NVRTC_SAFE_CALL(x)                                        \
  do {                                                            \
    nvrtcResult result = x;                                       \
    if (result != NVRTC_SUCCESS) {                                \
      std::cerr << "\nerror: " #x " failed with error "           \
                << nvrtcGetErrorString(result) << '\n';           \
      exit(1);                                                    \
    }                                                             \
  } while(0)
#define CUDA_SAFE_CALL(x)                                         \
  do {                                                            \
    CUresult result = x;                                          \
    if (result != CUDA_SUCCESS) {                                 \
      const char *msg;                                            \
      cuGetErrorName(result, &msg);                               \
      std::cerr << "\nerror: " #x " failed with error "           \
                << msg << '\n';                                   \
      exit(1);                                                    \
    }                                                             \
  } while(0)

/////////////////////////////////////////////////////////////////////
std::string ReplaceAll(std::string str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
    return str;
}

/////////////////////////////////////////////////////////////////////


static ex sfR_eval (const ex &x, const ex &y, const ex &x1, const ex &x2, const ex &x3, const ex &y1, const ex &y2, const ex &y3);
static ex sfR_deriv(const ex &x, const ex &y, const ex &x1, const ex &x2, const ex &x3, const ex &y1, const ex &y2, const ex &y3, unsigned diff_param);
static ex sfS_eval (const ex &x, const ex &y, const ex &x1, const ex &x2, const ex &x3, const ex &y1, const ex &y2, const ex &y3);
static ex sfS_deriv(const ex &x, const ex &y, const ex &x1, const ex &x2, const ex &x3, const ex &y1, const ex &y2, const ex &y3, unsigned diff_param);

DECLARE_FUNCTION_8P(sfR)
REGISTER_FUNCTION(sfR, eval_func(sfR_eval).
                             derivative_func(sfR_deriv).
                             latex_name("r"));
DECLARE_FUNCTION_8P(sfS)
REGISTER_FUNCTION(sfS, eval_func(sfS_eval).
                             derivative_func(sfS_deriv).
                             latex_name("s"));

ex sfR_eval (const ex &x, const ex &y, const ex &x1, const ex &x2, const ex &x3, const ex &y1, const ex &y2, const ex &y3) 
{
    return sfR(x,y,x1,x2,x3,y1,y2,y3).hold();
}
//    r_x = (y2-y3)/jac;
//    r_y = (x3-x2)/jac;
ex sfR_deriv(const ex &x, const ex &y, const ex &x1, const ex &x2, const ex &x3, const ex &y1, const ex &y2, const ex &y3, unsigned diff_param) 
{
  symbol r("r"), s("s");
  ex fx = x1*r + x2*s + x3*(1-r-s);
  ex fy = y1*r + y2*s + y3*(1-r-s);
  ex jac = fx.diff(r)*fy.diff(s) - fy.diff(r)*fx.diff(s);
  if(diff_param == 0)
      return (y2-y3)/jac;
  else if(diff_param == 1)
    return (x3-x2)/jac;
  else
    return 0;
}
ex sfS_eval (const ex &x, const ex &y, const ex &x1, const ex &x2, const ex &x3, const ex &y1, const ex &y2, const ex &y3) 
{
    return sfS(x,y,x1,x2,x3,y1,y2,y3).hold();
}
//    s_x = (y3-y1)/jac;
//    s_y = (x1-x3)/jac;
ex sfS_deriv(const ex &x, const ex &y, const ex &x1, const ex &x2, const ex &x3, const ex &y1, const ex &y2, const ex &y3, unsigned diff_param) 
{
  symbol r("r"), s("s");
  ex fx = x1*r + x2*s + x3*(1-r-s);
  ex fy = y1*r + y2*s + y3*(1-r-s);
  ex jac = fx.diff(r)*fy.diff(s) - fy.diff(r)*fx.diff(s);
  if(diff_param == 0)
      return (y3-y1)/jac;
  else if(diff_param == 1)
    return (x1-x3)/jac;
  else
    return 0;
}

/////////////////////////////////////////////////////
lst grad(ex &f, symbol &x, symbol &y) 
{
  return lst(f.diff(x), f.diff(y));
}

ex dot(lst l, lst r) 
{
  ex ret = 0;
  for (size_t i = 0; i < l.nops(); ++i)
       ret += l[i]*r[i];
  return ret;
}

//////////////////////////////////////////////////////
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

  void getNeighborNodes(int index, vector<int> &output)
  {
    set<int> nbSet;
    for(int i=0; i<elements.size(); i++)
    {
      Element *e = elements[i];
      for(int j=0; j<e->nodes.size(); j++)
      {
        if(e->nodes[j]->index == index) {
            for(int jj=0; jj<e->nodes.size(); jj++) {
              if(e->nodes[jj]->index != index)
                nbSet.insert(e->nodes[jj]->index);
            }
            break;
        }
      }
    }
    for (set<int>::iterator it=nbSet.begin(); it!=nbSet.end(); ++it) {
      output.push_back(*it);
    }
    sort(output.begin(), output.end());
  }

  /**
   * Return a list of neighbor nodes for every nodes
   */
  void getNeighborNodesList(int* lengthList, int maxLength, int *indexList)
  {
    vector< set<int>* > nbNodes;
    for(int i=0; i<nodes.size(); i++)
      nbNodes.push_back(new set<int>());

    for(int i=0; i<elements.size(); i++) {
      Element *e = elements[i];
      for(int j=0; j<e->nodes.size(); j++) {
        set<int> *nbSet = nbNodes[e->nodes[j]->index];
        for(int jj=0; jj<e->nodes.size(); jj++) {
            nbSet->insert(e->nodes[jj]->index);
        }
      }
    }

    for(int i=0; i<nodes.size(); i++)
    {
      vector<int> nb;
      set<int> *nbSet = nbNodes[i];
      for (set<int>::iterator it=nbSet->begin(); it!=nbSet->end(); ++it)
        nb.push_back(*it);
      sort(nb.begin(), nb.end());
      lengthList[i] = nb.size();
      for(int j=0; j<nb.size(); j++)
        indexList[i*maxLength + j] = nb[j];
      delete nbSet;
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

class FunctionSpace 
{
public:
  Mesh mesh;
  lst coords;     //x,y
  lst refCoords;  //r,s
  lst nodeCoords; //x1,x2,x3,y1,y2,y3
  lst sfRS;       //r(x,y), s(x,y)
  lst trans;
  FunctionSpace()
  {

  }
  FunctionSpace(Mesh &mesh, lst coords, string type, int order) 
  {
    this->mesh = mesh;

    this->coords = coords;
    
    ex x = coords[0];
    ex y = coords[1];

    symbol r("r"), s("s");
    refCoords = lst(r, s);

    symbol x1("x1"), x2("x2"), x3("x3");
    symbol y1("y1"), y2("y2"), y3("y3");
    nodeCoords = x1, x2, x3, y1, y2, y3;

    ex sfr = sfR(x,y,x1,x2,x3,y1,y2,y3);
    ex sfs = sfS(x,y,x1,x2,x3,y1,y2,y3);
    sfRS = lst(sfr, sfs);

    ex fx = x1*r + x2*s + x3*(1-r-s);
    ex fy = y1*r + y2*s + y3*(1-r-s);
    trans = lst(fx, fy);
  }

  lst getShapeFunctions() 
  {
    ex sfr = sfRS[0];
    ex sfs = sfRS[1];
    return lst(sfr, sfs, 1-sfr-sfs);
  }

  int getDOFNum()
  {
    return 3;
  }

  lst getTransform() 
  {
    return trans;
  }

  ex getJac() 
  {
    symbol r = ex_to<symbol>(refCoords[0]);
    symbol s = ex_to<symbol>(refCoords[1]);
    lst trans = getTransform();
    ex fx = trans[0];
    ex fy = trans[1];
    return fx.diff(r)*fy.diff(s) - fy.diff(r)*fx.diff(s);
  }
};

///////////////////////////////////////////////////////
class WeakForm 
{
public:
  FunctionSpace funcSpace;
  ex lhs[3][3];
  ex rhs[3];
  std::string codeTemplate;

  WeakForm(FunctionSpace &funcSpace, std::string &codeTemplate)
  {
    this->funcSpace = funcSpace;
    this->codeTemplate = codeTemplate;
  }

  void build(std::function<ex(ex,ex)> _lhs, std::function<ex(ex)> _rhs)
  {
    lst sfuncs = funcSpace.getShapeFunctions();
    ex sfr = sfuncs[0];
    ex sfs = sfuncs[1];

    int nDOFs = funcSpace.getDOFNum();
    ex x = funcSpace.coords[0];
    ex y = funcSpace.coords[1];
    lst trans = funcSpace.getTransform();
    ex fx = trans[0];
    ex fy = trans[1];

    ex jac = funcSpace.getJac();

    lst argsOrder;
    ex r = funcSpace.refCoords[0];
    ex s = funcSpace.refCoords[1];
    ex x1 = funcSpace.nodeCoords[0];
    ex x2 = funcSpace.nodeCoords[1];
    ex x3 = funcSpace.nodeCoords[2];
    ex y1 = funcSpace.nodeCoords[3];
    ex y2 = funcSpace.nodeCoords[4];
    ex y3 = funcSpace.nodeCoords[5];
    argsOrder = x1,x2,x3,y1,y2,y3,r,s;
    std::ostringstream oss;
    for(int j=0; j<nDOFs; j++) 
    {
      for(int i=0; i<nDOFs; i++) 
      {
        lhs[j][i] =  _lhs(sfuncs[j], sfuncs[i]).subs(lst(sfr==r, sfs==s)).subs(lst(x==fx, y==fy))*jac;
        //cout<<csrc_float<<lhs[j][i]<<endl;
        oss.str(""); oss.clear();
        oss<<csrc_float<<lhs[j][i]<<endl;
        //std::cout << oss.str().c_str();
        char keyWord[50];
        sprintf(keyWord, "$integrand%d%d$",i,j);
        codeTemplate = ReplaceAll(codeTemplate, std::string((char*)keyWord), std::string(oss.str().c_str()));
      }
      rhs[j] = _rhs(sfuncs[j]).subs(lst(sfr==r, sfs==s)).subs(lst(x==fx, y==fy))*jac;
      //cout<<csrc_float<<rhs[j]<<endl;
      oss.str(""); oss.clear();
      oss<<csrc_float<<rhs[j]<<endl;
      //std::cout << oss.str().c_str();
      //codeTemplate = ReplaceAll(codeTemplate, std::string(sprintf("$integrand%d%d$",???j)), std::string(oss.str().c_str()));
    }
    codeTemplate = ReplaceAll(codeTemplate, std::string("pow"), std::string("powf"));
    std::cout << codeTemplate << std::endl;

  }

  const char* getCode() {
    return this->codeTemplate.c_str();
  }

};

//////////////////////////////////////////////////////////////

#define MESH_W 1000L
#define MESH_H 100L

#define M (MESH_W+1)*(MESH_H+1) //size of matrix A M by N
#define N (MESH_W+1)*(MESH_H+1)
#define NE 2*MESH_W*MESH_H //number of elements

#define BLOCK_X 7 // number of integration points
#define BLOCK_Y 9 // number of expressions
#define BLOCK_Z ((int)(32*32)/(BLOCK_X*BLOCK_Y)) //number of elements in a block
#define NDOF 3 //number of DOFs
#define NNODE 3 //number of nodes
#define MAX_NEIGHBOR (2*NNODE+1)

std::string codeTemplate("                                           \n\
__constant__ float triW[7] = { 0.06296959f, 0.06619708f, 0.06296959f, 0.06619708f, 0.06296959f, 0.06619708f, 0.11250000f };         \n\
__constant__ float triR[7] = { 0.10128651f, 0.47014206f, 0.79742699f, 0.47014206f, 0.10128651f, 0.05971587f, 0.33333333f };         \n\
__constant__ float triS[7] = { 0.10128651f, 0.05971587f, 0.10128651f, 0.47014206f, 0.79742699f, 0.47014206f, 0.33333333f };         \n\
__constant__ float triT[7] = { 0.79742698f, 0.47014207f, 0.1012865f,  0.05971588f, 0.1012865f,  0.47014207f, 0.33333334f };         \n\
__device__ float integrand(int funIdx, float *params)                                                                               \n\
{                            \n\
  float x1 = params[0];      \n\
  float x2 = params[1];      \n\
  float x3 = params[2];      \n\
  float y1 = params[3];      \n\
  float y2 = params[4];      \n\
  float y3 = params[5];      \n\
  float r = params[6];       \n\
  float s = params[7];       \n\
  float t = params[8];       \n\
  if(funIdx == 0)            \n\
    return $integrand00$;    \n\
  if(funIdx == 1)            \n\
    return $integrand01$;    \n\
  if(funIdx == 2)            \n\
    return $integrand02$;    \n\
  if(funIdx == 3)            \n\
    return $integrand10$;    \n\
  if(funIdx == 4)            \n\
    return $integrand11$;    \n\
  if(funIdx == 5)            \n\
    return $integrand12$;    \n\
  if(funIdx == 6)            \n\
    return $integrand20$;    \n\
  if(funIdx == 7)            \n\
    return $integrand21$;    \n\
  if(funIdx == 8)            \n\
    return $integrand22$;    \n\
  return 0.0f;                                                                                                                             \n\
}                                                                                                                                          \n\
extern \"C\" __global__ void fea_kernel(float* A,                                                                                          \n\
    int *colA,                                                                                                                             \n\
    int *gNbrNodeLen, int *gNbrNodeIdx,                                                                                                    \n\
    float *X, float *Y,                                                                                                                    \n\
    int *gIdx                                                                                                                              \n\
  )                                                                                                                                        \n\
{                                                                                                                                          \n\
  __shared__ float sX[BLOCK_Z*NNODE];    //shared memory of copy of X in the current block                                                 \n\
  __shared__ float sY[BLOCK_Z*NNODE];    //shared memory of copy of Y in the current block                                                 \n\
  __shared__ float sGIdx[BLOCK_Z*NNODE]; //shared memory of copy of gIdx in the current block                                              \n\
                                                                                                                                           \n\
  int gEleIdx = BLOCK_Z*blockIdx.x + threadIdx.z; //global element index                                                                   \n\
  int sEleIdx = NNODE*threadIdx.z;                //global element index in shared memory arrays: sX,sY,sGIdx                              \n\
                                                                                                                                           \n\
  // copy from global memory to shared memory for X, Y and gIdx                                                                            \n\
  if(threadIdx.x==0 && threadIdx.y==0)                                                                                                     \n\
  {                                                                                                                                        \n\
    for(int i=0; i<NNODE; i++)                                                                                                             \n\
      sX[sEleIdx+i]=X[NNODE*gEleIdx+i];                                                                                                    \n\
                                                                                                                                           \n\
    for(int i=0; i<NNODE; i++)                                                                                                             \n\
      sY[sEleIdx+i]=Y[NNODE*gEleIdx+i];                                                                                                    \n\
                                                                                                                                           \n\
   for(int i=0; i<NNODE; i++)                                                                                                              \n\
      sGIdx[sEleIdx+i]=gIdx[NNODE*gEleIdx+i];                                                                                              \n\
  }                                                                                                                                        \n\
  __syncthreads();                                                                                                                         \n\
                                                                                                                                           \n\
  //local matrix row and column index                                                                                                      \n\
  //threadIdx.y = 0,1,2,3,4,5,6,7,8 (BLOCK_Y)                                                                                              \n\
  int li = threadIdx.y / NDOF;                                                                                                             \n\
  int lj = threadIdx.y % NDOF;                                                                                                             \n\
  __shared__ float localFlatMatrix[BLOCK_Y*BLOCK_Z]; //array for the local flat matrices of all the elememnts in the current block         \n\
  for(int i=0; i<BLOCK_Y*BLOCK_Z; i++) localFlatMatrix[i] = 0.0f;                                                                          \n\
  int lfmIdx = threadIdx.z*BLOCK_Y + threadIdx.y; //local flat matrix index of the integrand of threadIdx.y                                \n\
  float params[3*NNODE]; //parameters array of integrand                                                                                   \n\
                                                                                                                                           \n\
  //compute local matrix                                                                                                                   \n\
  if(gEleIdx < NE)                                                                                                                         \n\
  {                                                                                                                                        \n\
    for(int i=0; i<NNODE; i++)                                                                                                             \n\
      params[i] = sX[sEleIdx+i];                                                                                                           \n\
                                                                                                                                           \n\
    for(int i=0; i<NNODE; i++)                                                                                                             \n\
      params[NNODE+i] = sY[sEleIdx+i];                                                                                                     \n\
                                                                                                                                           \n\
    params[2*NNODE+0] = triR[threadIdx.x];                                                                                                 \n\
    params[2*NNODE+1] = triS[threadIdx.x];                                                                                                 \n\
    params[2*NNODE+2] = triT[threadIdx.x]; //triT[threadIdx.x]=1.0-triR[threadIdx.x]-triS[threadIdx.x];                                    \n\
                                                                                                                                           \n\
    atomicAdd( &localFlatMatrix[lfmIdx], triW[threadIdx.x]*integrand(threadIdx.y, params) );                                               \n\
  }                                                                                                                                        \n\
  __syncthreads();                                                                                                                         \n\
                                                                                                                                           \n\
  //write to gobal matrix A                                                                                                                \n\
  if(gEleIdx < NE)                                                                                                                         \n\
  {                                                                                                                                        \n\
    if(threadIdx.x == 0)                                                                                                                   \n\
    {                                                                                                                                      \n\
      //global matrix row and column index                                                                                                 \n\
      int gi  = sGIdx[sEleIdx + li];                                                                                                       \n\
      int gj  = sGIdx[sEleIdx + lj];                                                                                                       \n\
      //binary search for colIdx?                                                                                                           \n\
      int colIdx = 0;                                                                                                                      \n\
      for(colIdx = 0; colIdx<gNbrNodeLen[gi]; colIdx++)                                                                                    \n\
      {                                                                                                                                   \n\
        if(gj == gNbrNodeIdx[gi*MAX_NEIGHBOR + colIdx]) break;                                                                             \n\
      }                                                                                                                                   \n\
      colA[gi*MAX_NEIGHBOR + colIdx] = gj;                                                                                                 \n\
      atomicAdd( &A[gi*MAX_NEIGHBOR + colIdx], localFlatMatrix[lfmIdx] );                                                                  \n\
    }                                                                                                                                      \n\
  }                                                                                                                                        \n\
}                                                                                                                                          \n\
                                                                                                                                           \n");

//////////////////////////////////////////////////////////////
/*
export CUDA_PATH=/usr/local/cuda-9.1
g++ --std=c++11 fea_symbolic_nvrtc_sparse2.cpp -o fea_symbolic_nvrtc_sparse2 -I $CUDA_PATH/include -L $CUDA_PATH/lib64 -lginac -lnvrtc -lcuda -Wl,-rpath,$CUDA_PATH/lib64
*/
int main()
{

  RectangleMesh mesh(-3.0, 3.0, -3.0, 3.0, MESH_W, MESH_H);
  cout << "Matrix A dim (M, N) = (" << M << ", " << N << ")" << endl;
  cout << "Number of nodes = " << mesh.nodes.size() << endl;
  //mesh.printMesh();

  symbol x("x"), y("y");
  ex f = -2*(x*x + y*y) + 36; //Right hand side(RHS)

  FunctionSpace fs = FunctionSpace(mesh, lst(x, y), "Lagrange", 1);

  WeakForm wf(fs, codeTemplate);
  wf.build(
    [&](ex u, ex v) { return dot(grad(u,x,y), grad(v,x,y)); },
    [&](ex v) { return f*v; }
  );

  // Create an instance of nvrtcProgram with the SAXPY code string.
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
    nvrtcCreateProgram(&prog,         // prog
                       wf.getCode(),         // buffer
                       "code.cu",    // name
                       0,             // numHeaders
                       NULL,          // headers
                       NULL));        // includeNames
//cout << (std::string("--define-macro=MESH_W=")+std::to_string(MESH_W)).c_str() << endl;
  // Compile the program for compute_30 with fmad disabled.
  const char *opts[] = {"--gpu-architecture=compute_30",
                        "--define-macro=MESH_W=1000",
                        "--define-macro=MESH_H=100",
                        //(std::string("--define-macro=MESH_W=")+std::to_string(MESH_W)).c_str(),
                        //(std::string("--define-macro=MESH_H=")+std::to_string(MESH_H)).c_str(),
                        "--define-macro=M=(MESH_W+1)*(MESH_H+1)",
                        "--define-macro=N=(MESH_W+1)*(MESH_H+1)",
                        "--define-macro=NE=2*MESH_W*MESH_H",
                        "--define-macro=BLOCK_X=7",
                        "--define-macro=BLOCK_Y=9",
                        "--define-macro=BLOCK_Z=((int)(32*32)/(BLOCK_X*BLOCK_Y))",
                        "--define-macro=NDOF=3",
                        "--define-macro=NNODE=3",
                        "--define-macro=MAX_NEIGHBOR=(2*NNODE+1)",
                        "--fmad=false" };
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                  13,     // numOptions
                                                  opts); // options
  // Obtain compilation log from the program.
  size_t logSize;
  NVRTC_SAFE_CALL(nvrtcGetProgramLogSize(prog, &logSize));
  char *log = new char[logSize];
  NVRTC_SAFE_CALL(nvrtcGetProgramLog(prog, log));
  std::cout << log << '\n';
  delete[] log;
  if (compileResult != NVRTC_SUCCESS) {
    exit(1);
  }
  // Obtain PTX from the program.
  size_t ptxSize;
  NVRTC_SAFE_CALL(nvrtcGetPTXSize(prog, &ptxSize));
  char *ptx = new char[ptxSize];
  //std::cout<<ptx<<std::endl;
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));
  // Destroy the program.
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  // Load the generated PTX and get a handle to the SAXPY kernel.
  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  CUfunction kernel;
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, "fea_kernel"));

// Generate input for execution, and create output buffers.
  float *X  = new float[ NE*NNODE ];
  float *Y  = new float[ NE*NNODE ];
  int *gIdx = new int[ NE*NNODE ];
  int *colA = new int[ M*MAX_NEIGHBOR];
  float *A  = new float[ M*MAX_NEIGHBOR ];
  int *gNbrNodeLen = new int[ M];
  int *gNbrNodeIdx = new int[ M*MAX_NEIGHBOR ];

  for(size_t i=0; i<M*MAX_NEIGHBOR; i++) A[i] = 0.0f;
  for(size_t i=0; i<M; i++) gNbrNodeLen[i] = 0;
  for(size_t i=0; i<M*MAX_NEIGHBOR; i++) gNbrNodeIdx[i] = 0;

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
  mesh.getNeighborNodesList(gNbrNodeLen, MAX_NEIGHBOR, gNbrNodeIdx);
  //for(size_t i=0; i<M; i++) {
  //  cout << gNbrNodeLen[i] << " -> ";
  //  for(size_t j=0; j<gNbrNodeLen[i]; j++)
  //    cout << gNbrNodeIdx[i*MAX_NEIGHBOR+j] << " ";
  //  cout << endl;
  //}

  CUdeviceptr dA, dColA, dX, dY, dGIdx, dNbrNodeLen, dNbrNodeIdx;

  clock_t tStart, tCUMemAlloc, tMemHtoD, tMemDtoH;
  tStart = clock();
  CUDA_SAFE_CALL(cuMemAlloc(&dX, NE*NNODE*sizeof(float)));
  CUDA_SAFE_CALL(cuMemAlloc(&dY, NE*NNODE*sizeof(float)));
  CUDA_SAFE_CALL(cuMemAlloc(&dGIdx, NE*NNODE*sizeof(int)));
  CUDA_SAFE_CALL(cuMemAlloc(&dA, M*MAX_NEIGHBOR*sizeof(float)));
  CUDA_SAFE_CALL(cuMemAlloc(&dColA, M*MAX_NEIGHBOR*sizeof(float)));
  CUDA_SAFE_CALL(cuMemAlloc(&dNbrNodeLen, M*sizeof(int)));
  CUDA_SAFE_CALL(cuMemAlloc(&dNbrNodeIdx, M*MAX_NEIGHBOR*sizeof(int)));

  tCUMemAlloc = clock() - tStart;

  tStart = clock();
  CUDA_SAFE_CALL(cuMemcpyHtoD(dA, A, M*MAX_NEIGHBOR*sizeof(float)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dX, X, NE*NNODE*sizeof(float)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dY, Y, NE*NNODE*sizeof(float)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dGIdx, gIdx, NE*NNODE*sizeof(int)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dNbrNodeLen, gNbrNodeLen, M*sizeof(int)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dNbrNodeIdx, gNbrNodeIdx, M*MAX_NEIGHBOR*sizeof(int)));
  tMemHtoD = clock() - tStart;

  // Execute kernal.
  void *args[] = { &dA, &dColA, &dNbrNodeLen, &dNbrNodeIdx, &dX, &dY, &dGIdx};
  CUevent start, stop;
  float elapsed = 0;
  CUDA_SAFE_CALL(cuEventCreate(&start,CU_EVENT_DEFAULT));
  CUDA_SAFE_CALL(cuEventCreate(&stop,CU_EVENT_DEFAULT));
  CUDA_SAFE_CALL(cuEventRecord(start, 0));
  CUDA_SAFE_CALL(
    cuLaunchKernel(kernel,
                   (NE+BLOCK_Z-1)/BLOCK_Z, 1, 1,    // grid dim
                   BLOCK_X, BLOCK_Y, BLOCK_Z,   // block dim
                   0, NULL,             // shared mem and stream
                   args, 0));           // arguments
  CUDA_SAFE_CALL(cuEventRecord(stop, 0));
  CUDA_SAFE_CALL(cuEventSynchronize(stop));
  CUDA_SAFE_CALL(cuEventElapsedTime(&elapsed, start, stop));
  CUDA_SAFE_CALL(cuCtxSynchronize());

  // Retrieve and print output.
  tStart = clock();
  CUDA_SAFE_CALL(cuMemcpyDtoH(A, dA, M*MAX_NEIGHBOR*sizeof(float)));
  CUDA_SAFE_CALL(cuMemcpyDtoH(colA, dColA, M*MAX_NEIGHBOR*sizeof(float)));
  tMemDtoH = clock() - tStart;

  std::cout << "mesh = " << MESH_W << "*" << MESH_H << endl;
  std::cout << "grid dim.x = " << (NE+BLOCK_Z-1)/BLOCK_Z << endl;
  std::cout << "GPU Time: " << elapsed << "ms" << std::endl;
  std::cout << "Time cuMemAlloc = " << tCUMemAlloc/1000.0 << "ms" << std::endl;
  std::cout << "Time HtoD = " << tMemHtoD/1000.0 << "ms" << std::endl;
  std::cout << "Time DtoH = " << tMemDtoH/1000.0 << "ms" << std::endl;
  for(size_t i=0; i<16 && i<M; i++) {
    for(size_t j=0; j<gNbrNodeLen[i]; j++) {
      std::cout << "(" << i << "," << colA[i*MAX_NEIGHBOR+j] << ")" << " " << A[i*MAX_NEIGHBOR+j] << "    ";
    }
    cout << endl;
  }

  // Release resources.
  CUDA_SAFE_CALL(cuMemFree(dNbrNodeLen));
  CUDA_SAFE_CALL(cuMemFree(dNbrNodeIdx));
  CUDA_SAFE_CALL(cuMemFree(dA));
  CUDA_SAFE_CALL(cuMemFree(dColA));
  CUDA_SAFE_CALL(cuMemFree(dX));
  CUDA_SAFE_CALL(cuMemFree(dY));
  CUDA_SAFE_CALL(cuMemFree(dGIdx));
  CUDA_SAFE_CALL(cuModuleUnload(module));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
  delete[] A;
  delete[] colA;
  delete[] gNbrNodeLen;
  delete[] gNbrNodeIdx;
  delete[] X;
  delete[] Y;
  delete[] gIdx;

  return 0;
}

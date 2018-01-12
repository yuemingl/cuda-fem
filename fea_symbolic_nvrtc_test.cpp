#include <iostream>
#include <ctime>
#include <vector>
#include <ginac/ginac.h>
#include <chrono>
#include <sstream> // std::ostringstream

#include <nvrtc.h>
#include <cuda.h>

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

  WeakForm(FunctionSpace &funcSpace) 
  {
    this->funcSpace = funcSpace;
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
        std::cout << oss.str().c_str();
      }
      rhs[j] = _rhs(sfuncs[j]).subs(lst(sfr==r, sfs==s)).subs(lst(x==fx, y==fy))*jac;
      //cout<<csrc_float<<rhs[j]<<endl;
      oss.str(""); oss.clear();
      oss<<csrc_float<<rhs[j]<<endl;
      std::cout << oss.str().c_str();
    }
  }

};

//////////////////////////////////////////////////////////////

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

const char *code = "                                           \n\
extern \"C\" __global__ void fea_kernel(float* A,                                                                                                       \n\
    float *X, float *Y,                                                                                                                    \n\
    int *gIdx                                                                                                                              \n\
  )                                                                                                                                        \n\
{                                                                                                                                          \n\
}                                                                                                                                          \n\
                                                                                                                                           \n";


//////////////////////////////////////////////////////////////
/*
export CUDA_PATH=/usr/local/cuda-9.1
g++ --std=c++11 fea_symbolic_nvrtc.cpp -o fea_symbolic_nvrtc -I $CUDA_PATH/include -L $CUDA_PATH/lib64 -lnvrtc -lcuda -Wl,-rpath,$CUDA_PATH/lib64 -lginac
*/
int main()
{
  // Create an instance of nvrtcProgram with the SAXPY code string.
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(
    nvrtcCreateProgram(&prog,         // prog
                       code,         // buffer
                       "code.cu",    // name
                       0,             // numHeaders
                       NULL,          // headers
                       NULL));        // includeNames
  // Compile the program for compute_30 with fmad disabled.
  const char *opts[] = {"--gpu-architecture=compute_30",
                        "--fmad=false"};
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                  2,     // numOptions
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
  RectangleMesh mesh(-3.0, 3.0, -3.0, 3.0, MESH_W, MESH_H);
  mesh.printMesh();

  symbol x("x"), y("y");
  ex f = -2*(x*x + y*y) + 36; //Right hand side(RHS)

  FunctionSpace fs = FunctionSpace(mesh, lst(x, y), "Lagrange", 1);

  WeakForm wf(fs);
  wf.build(
    [&](ex u, ex v) { return dot(grad(u,x,y), grad(v,x,y)); },
    [&](ex v) { return f*v; }
  );

  float *A  = new float[ M*N*sizeof(float) ];
  float *X  = new float[ NE*NNODE*sizeof(float) ];
  float *Y  = new float[ NE*NNODE*sizeof(float) ];
  int *gIdx = new int[ NE*NNODE*sizeof(int) ];

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

  CUdeviceptr dA, dX, dY, dGIdx;
  CUDA_SAFE_CALL(cuMemAlloc(&dA, M*N*sizeof(float)));
  CUDA_SAFE_CALL(cuMemAlloc(&dX, NE*NNODE*sizeof(float)));
  CUDA_SAFE_CALL(cuMemAlloc(&dY, NE*NNODE*sizeof(float)));
  CUDA_SAFE_CALL(cuMemAlloc(&dGIdx, NE*NNODE*sizeof(int)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dX, X, NE*NNODE*sizeof(float)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dY, Y, NE*NNODE*sizeof(float)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dGIdx, gIdx, NE*NNODE*sizeof(int)));

  // Execute kernal.
  void *args[] = { &dA, &dX, &dY, &dGIdx};
  CUDA_SAFE_CALL(
    cuLaunchKernel(kernel,
                   2, 1, 1,    // grid dim
                   BLOCK_X, BLOCK_Y, BLOCK_Z,   // block dim
                   0, NULL,             // shared mem and stream
                   args, 0));           // arguments
  CUDA_SAFE_CALL(cuCtxSynchronize());

  // Retrieve and print output.
  CUDA_SAFE_CALL(cuMemcpyDtoH(A, dA, M*N*sizeof(float)));
  for(size_t i=0; i<M; i++) {
    for(size_t j=0; j<N; j++) {
      std::cout << A[i*N+j] << " ";
    }
    std::cout << std::endl;
  }

  // Release resources.
  CUDA_SAFE_CALL(cuMemFree(dA));
  CUDA_SAFE_CALL(cuMemFree(dX));
  CUDA_SAFE_CALL(cuMemFree(dY));
  CUDA_SAFE_CALL(cuMemFree(dGIdx));
  CUDA_SAFE_CALL(cuModuleUnload(module));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
  delete[] A;
  delete[] X;
  delete[] Y;
  delete[] gIdx;
  return 0;
}

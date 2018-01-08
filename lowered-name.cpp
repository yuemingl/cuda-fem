#include <nvrtc.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <string>

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

const char *gpu_program = "                                     \n\
static __global__ void f1(int *result) { *result = 10; }        \n\
namespace N1 {                                                  \n\
  namespace N2 {                                                \n\
    __global__ void f2(int *result) { *result = 20; }           \n\
  }                                                             \n\
}                                                               \n\
template<typename T>                                            \n\
__global__ void f3(int *result) { *result = sizeof(T); }        \n\
                                                                \n";
/*
export CUDA_PATH=/usr/local/cuda-9.1
g++ lowered-name.cpp -o lowered-name -I $CUDA_PATH/include -L $CUDA_PATH/lib64 -lnvrtc -lcuda -Wl,-rpath,$CUDA_PATH/lib64
*/
int main()
{
  // Create an instance of nvrtcProgram
  nvrtcProgram prog;
  NVRTC_SAFE_CALL(nvrtcCreateProgram(&prog,         // prog
                                     gpu_program,   // buffer
                                     "prog.cu",     // name
                                     0,             // numHeaders
                                     NULL,          // headers
                                     NULL));        // includeNames

  // add all name expressions for kernels
  std::vector<std::string> name_vec;
  std::vector<int> expected_result;
  
  // note the name expressions are parsed as constant expressions
  name_vec.push_back("&f1");
  expected_result.push_back(10);
  
  name_vec.push_back("N1::N2::f2");
  expected_result.push_back(20);
    
  name_vec.push_back("f3<int>");
  expected_result.push_back(sizeof(int));
  
  name_vec.push_back("f3<double>");
  expected_result.push_back(sizeof(double));
  
  // add name expressions to NVRTC. Note this must be done before
  // the program is compiled.
  for (size_t i = 0; i < name_vec.size(); ++i)
    NVRTC_SAFE_CALL(nvrtcAddNameExpression(prog, name_vec[i].c_str()));
  
  nvrtcResult compileResult = nvrtcCompileProgram(prog,  // prog
                                                  0,     // numOptions
                                                  NULL); // options
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
  NVRTC_SAFE_CALL(nvrtcGetPTX(prog, ptx));

  // Load the generated PTX
  CUdevice cuDevice;
  CUcontext context;
  CUmodule module;
  
  CUDA_SAFE_CALL(cuInit(0));
  CUDA_SAFE_CALL(cuDeviceGet(&cuDevice, 0));
  CUDA_SAFE_CALL(cuCtxCreate(&context, 0, cuDevice));
  CUDA_SAFE_CALL(cuModuleLoadDataEx(&module, ptx, 0, 0, 0));
  
  CUdeviceptr dResult;
  int hResult = 0;
  CUDA_SAFE_CALL(cuMemAlloc(&dResult, sizeof(hResult)));
  CUDA_SAFE_CALL(cuMemcpyHtoD(dResult, &hResult, sizeof(hResult)));
  
  // for each of the name expressions previously provided to NVRTC,
  // extract the lowered name for corresponding __global__ function,
  // and launch it.
  
  for (size_t i = 0; i < name_vec.size(); ++i) {
    const char *name;
    
    // note: this call must be made after NVRTC program has been 
    // compiled and before it has been destroyed.
    NVRTC_SAFE_CALL(nvrtcGetLoweredName(
                          prog, 
			  name_vec[i].c_str(), // name expression
			  &name                // lowered name
                                        ));
    
    // get pointer to kernel from loaded PTX
    CUfunction kernel;
    CUDA_SAFE_CALL(cuModuleGetFunction(&kernel, module, name));
    
    // launch the kernel
    std::cout << "\nlaunching " << name << " ("
	      << name_vec[i] << ")" << std::endl;
    
    void *args[] = { &dResult };
    CUDA_SAFE_CALL(
      cuLaunchKernel(kernel,
		     1, 1, 1,             // grid dim
		     1, 1, 1,             // block dim
		     0, NULL,             // shared mem and stream
		     args, 0));           // arguments
    CUDA_SAFE_CALL(cuCtxSynchronize());
    
    // Retrieve the result
    CUDA_SAFE_CALL(cuMemcpyDtoH(&hResult, dResult, sizeof(hResult)));
    
    // check against expected value
    if (expected_result[i] != hResult) {
      std::cout << "\n Error: expected result = " << expected_result[i]
		<< " , actual result = " << hResult << std::endl;
      exit(1);
    }
  }  // for
    
  // Release resources.
  CUDA_SAFE_CALL(cuMemFree(dResult));
  CUDA_SAFE_CALL(cuModuleUnload(module));
  CUDA_SAFE_CALL(cuCtxDestroy(context));
  
  // Destroy the program. 
  NVRTC_SAFE_CALL(nvrtcDestroyProgram(&prog));

  return 0;
}


#include <iostream>
#include <chrono>

#include <hip/hip_runtime_api.h>
#include <miopen/miopen.h>

#define CHECK_HIP(cmd) \
{\
    hipError_t hip_error  = cmd;\
    if (hip_error != hipSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(hip_error), hip_error,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
	}\
}

const char* mio_err[] = {
    "StatusSuccess        ",
    "StatusNotInitialized ",
    "StatusInvalidValue   ",
    "StatusBadParm        ",
    "StatusAllocFailed    ",
    "StatusInternalError  ",
    "StatusNotImplemented ",
    "StatusUnknownError   "
};

#define INFO(msg) std::cerr << msg << std::endl;

#define CHECK_MIO(cmd) \
{\
    miopenStatus_t miostat = cmd;\
    if (miostat != miopenStatusSuccess) { \
      fprintf(stderr, "error: '%s'(%d) at %s:%d\n", mio_err[(int)miostat], miostat,__FILE__, __LINE__); \
    exit(EXIT_FAILURE);\
	}\
}


/// support only float32 for now
struct TensorDesc {
    miopenTensorDescriptor_t desc;
    int n,c,h,w;

    TensorDesc(int n, int c, int h, int w) : n(n), c(c), h(h), w(w) {
	CHECK_MIO(miopenCreateTensorDescriptor(&desc));
	CHECK_MIO(miopenSet4dTensorDescriptor(desc, miopenFloat, n, c, h, w));
    }

    ~TensorDesc() {
	CHECK_MIO(miopenDestroyTensorDescriptor(desc));
    }

    std::ostream& print_dims(std::ostream& os) const {
	return os << "(" << n << "," << c << "," << h << "," << h << ")";
    }
};

struct Tensor : public TensorDesc {
    void* data;
    size_t data_size;
    Tensor(int n, int c, int h, int w) : TensorDesc(n, c, h, w) {
	data_size = n;
	data_size *= c; data_size *= h; data_size *= w; data_size *= 4;
	INFO("Allocating Float Tensor (" << n << "," << c << "," << h << "," << h << "), total size: " << data_size / 1024 << " kB");
	CHECK_HIP(hipMalloc(&data, data_size));

    }

    ~Tensor() {
	CHECK_HIP(hipFree(data));
    }
};

struct Convolution {
    miopenConvolutionDescriptor_t desc;

    Convolution(int pad_h, int pad_w, int u, int v, int upscalex, int upscaley) {
	CHECK_MIO(miopenCreateConvolutionDescriptor(&desc));
	CHECK_MIO(miopenInitConvolutionDescriptor(desc, miopenConvolution, pad_h, pad_w, u, v, upscalex, upscaley));
    }

    ~Convolution() {
	CHECK_MIO(miopenDestroyConvolutionDescriptor(desc));
    }
};


int main(int argc, char *argv[])
{

    hipDeviceProp_t props;
    CHECK_HIP(hipGetDeviceProperties(&props, 0/*deviceID*/));
    printf ("info: running on device %s\n", props.name);
    #ifdef __HIP_PLATFORM_HCC__
      printf ("info: architecture on AMD GPU device is: %d\n",props.gcnArch);
    #endif

    miopenHandle_t mio_handle;
    miopenCreate(&mio_handle);

    // batch_size, channels, w, h
    Tensor t1(128, 384, 13, 13);
    Convolution conv(0, 0, 1, 1, 1, 1); // TODO: check what the benchmark maps these to
    // channels_out, channels_in, filter size kx x ky
    Tensor weights(384, 384, 3, 3);

    int n,c,h,w;
    CHECK_MIO(miopenGetConvolutionForwardOutputDim(conv.desc, t1.desc, weights.desc, &n, &c, &h, &w));
    INFO("Output dim tensor: " << n << ", " << c << ", " << h << ", " << w);

    Tensor tout(n, c, h, w);

    size_t workspace_size;
    CHECK_MIO(miopenConvolutionForwardGetWorkSpaceSize(mio_handle, weights.desc, t1.desc, conv.desc, tout.desc, &workspace_size));

    INFO("Workspace size required for convolution: " << workspace_size);
    void* workspace;
    CHECK_HIP(hipMalloc(&workspace, workspace_size))

    // TODO: find best algo, and benchmark!
    miopenConvAlgoPerf_t perfs[10];
    int returned_algos;
    CHECK_MIO(miopenFindConvolutionForwardAlgorithm(mio_handle, t1.desc, t1.data, weights.desc, weights.data, conv.desc, tout.desc, tout.data, 10, &returned_algos, perfs, workspace, workspace_size, false));

    INFO("found " << returned_algos << " fwd algorithms: ");
    for (int i = 0; i < returned_algos; ++i) {
	INFO("FWD Algo: " << perfs[i].fwd_algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory)
    }

    float alpha = 1.f;
    float beta = 1.f;
    CHECK_HIP(hipDeviceSynchronize());
    auto tic = std::chrono::steady_clock::now();
    for (int i = 0; i < 10; ++i) {
	CHECK_MIO(miopenConvolutionForward(mio_handle, &alpha, t1.desc, t1.data, weights.desc, weights.data, conv.desc, perfs[0].fwd_algo, &beta, tout.desc, tout.data, workspace, workspace_size));
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto toc = std::chrono::steady_clock::now();
    INFO("Time for FWD: " << std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count() << " ms");

    miopenDestroy(mio_handle);
    return 0;
}

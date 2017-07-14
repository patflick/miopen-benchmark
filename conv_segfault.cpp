#include <hip/hip_runtime_api.h>
#include <miopen/miopen.h>

#include <stdio.h>
#include <iostream>

#define CHECK_HIP(cmd) \
{\
    hipError_t hip_error  = cmd;\
    if (hip_error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(hip_error), hip_error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}


#define CHECK_MIO(cmd) \
{\
    miopenStatus_t miostat = cmd;\
    if (miostat != miopenStatusSuccess) { \
        fprintf(stderr, " MIOpen error (%d) at %s:%d\n", miostat,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}


struct Tensor {
    miopenTensorDescriptor_t desc;
    void* data;
    size_t data_size;
    Tensor(int n, int c, int h, int w) {
        CHECK_MIO(miopenCreateTensorDescriptor(&desc));
        CHECK_MIO(miopenSet4dTensorDescriptor(desc, miopenFloat, n, c, h, w));
        data_size = n*c*h*w*sizeof(float);
        CHECK_HIP(hipMalloc(&data, data_size));
    }
};

int main(int argc, char *argv[])
{
    int devcount;
    CHECK_HIP(hipGetDeviceCount(&devcount));
    std::cout << "Number of HIP devices found: " << devcount << std::endl;
    if (devcount <= 0)
        exit(EXIT_FAILURE);

    miopenHandle_t mio_handle;
    CHECK_MIO(miopenCreate(&mio_handle));

    /* create conv desc */
    miopenConvolutionDescriptor_t convdesc;
    CHECK_MIO(miopenCreateConvolutionDescriptor(&convdesc));
    CHECK_MIO(miopenInitConvolutionDescriptor(convdesc, miopenConvolution, 1, 1, 1, 1, 1, 1));

    // create input, output and weights tensors
    Tensor input(128, 3, 32, 32);
    Tensor output(128, 64, 32, 32);
    Tensor weights(64, 3, 3, 3);

    // create workspace
    size_t workspace_size;
    void* workspace;
    CHECK_MIO(miopenConvolutionForwardGetWorkSpaceSize(mio_handle, weights.desc, input.desc, convdesc, output.desc, &workspace_size));
    CHECK_HIP(hipMalloc(&workspace, workspace_size));

    // find best algo, and benchmark!
    miopenConvAlgoPerf_t perfs[4];
    int returned_algos;
    CHECK_MIO(miopenFindConvolutionForwardAlgorithm(mio_handle, input.desc, input.data, weights.desc, weights.data, convdesc, output.desc, output.data, 4, &returned_algos, perfs, workspace, workspace_size, false));
    return 0;
}

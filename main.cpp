
#include <assert.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>


#include <miopen/miopen.h>


//#define WITH_CL

#ifdef WITH_CL
#include <CL/cl.hpp>
#define device_mem_t cl_mem

struct ClHandle {
    static std::vector<cl::Platform> getPlatforms() {
        std::vector<cl::Platform> plats;
        cl::Platform::get(&plats);
        return plats;
    }

    static std::vector<cl::Device> getDevices(const std::vector<cl::Platform>& platforms) {
        std::vector<cl::Device> devs;
        platforms[0].getDevices(CL_DEVICE_TYPE_ALL, &devs);
        return devs;
    }

    static cl_context get_handle() {
        static std::vector<cl::Platform> platforms = getPlatforms();
        static std::vector<cl::Device> devices = getDevices(platforms);
        static cl::Context ctx = cl::Context(devices[0]);
        return ctx();
    }
};

device_mem_t device_alloc(size_t size) {
    cl_int err;
    cl_mem buf = clCreateBuffer(ClHandle::get_handle(), CL_MEM_READ_WRITE, size, NULL, &err);
    if (err) {
        fprintf(stderr, "error: opencl couldn't allocate buffer, error code: %i", err);
        exit(EXIT_FAILURE);
    }
    return buf;
}

void device_free(device_mem_t m) {
    cl_int err = clReleaseMemObject(m);
    if (err) {
        fprintf(stderr, "error: opencl couldn't allocate buffer");
        exit(EXIT_FAILURE);
    }
}

void device_init() {
}

#define CHECK_HIP(cmd) 

#else
#include <hip/hip_runtime_api.h>

#define CHECK_HIP(cmd) \
{\
    hipError_t hip_error  = cmd;\
    if (hip_error != hipSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", hipGetErrorString(hip_error), hip_error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

#define device_mem_t void*

device_mem_t device_alloc(size_t size) {
    void* ptr;
    CHECK_HIP(hipMalloc(&ptr, size));
    return ptr;
}

void device_free(device_mem_t m) {
    CHECK_HIP(hipFree(m));
}

void device_init() {
    int devcount;
    CHECK_HIP(hipGetDeviceCount(&devcount));
    std::cout << "Number of HIP devices found: " << devcount << std::endl;

    for (int d = 0; d < devcount; ++d) {
        hipDeviceProp_t props;
        CHECK_HIP(hipGetDeviceProperties(&props, d/*deviceID*/));
        std::cout << d << ": Device " << props.name << std::endl;
        std::cout << "\t\tGMem:\t" << props.totalGlobalMem/1024/1024 << " MiB" << std::endl;
        std::cout << "\t\twarps:\t" << props.warpSize << std::endl;
        std::cout << "\t\tCUs:\t" << props.multiProcessorCount << std::endl;
#ifdef __HIP_PLATFORM_HCC__
        std::cout << "\t\tArch:\t" << props.gcnArch << std::endl;
#endif
    }
}

#endif

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

#define DEBUG(msg) std::cerr << msg << std::endl;

#define CHECK_MIO(cmd) \
{\
    miopenStatus_t miostat = cmd;\
    if (miostat != miopenStatusSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", mio_err[(int)miostat], miostat,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

struct mio {
private:
    static miopenHandle_t get_handle() {
        miopenHandle_t h;
        CHECK_MIO(miopenCreate(&h));
        return h;
    }
public:
    static miopenHandle_t handle() {
        static miopenHandle_t h = get_handle();
        return h;
    }
};

struct DevBuffer {
    device_mem_t data;
    size_t size;

    DevBuffer() : data(NULL), size(0) {}

    DevBuffer(size_t size) : size(size) {
        data = device_alloc(size);
    }

    DevBuffer(const DevBuffer& o) = delete;
    DevBuffer(DevBuffer&& o) {
        this->data = o.data; o.data = NULL;
        this->size = o.size; o.size = 0;
    }

    DevBuffer& operator=(const DevBuffer& o) = delete;
    DevBuffer& operator=(DevBuffer&& o) {
        this->free();
        this->data = o.data; o.data = NULL;
        this->size = o.size; o.size = 0;
        return *this;
    }

    void free() {
        if (data != NULL && size > 0) {
            device_free(data);
            data = NULL;
            size = 0;
        }
    }

    ~DevBuffer() {
        free();
    }
};

// 4D Dimensions as NCHW
struct Dim {
    int n;
    int c;
    int h;
    int w;

    Dim() : n(0), c(0), h(0), w(0) {}
    Dim(int n, int c, int h, int w) : n(n), c(c), h(h), w(h) {}
    Dim(const Dim&) = default;
    Dim(Dim&&) = default;
    Dim& operator=(const Dim&) = default;
    Dim& operator=(Dim&&) = default;
};


/// support only float32 for now
struct TensorDesc : public Dim {
    miopenTensorDescriptor_t desc;

    TensorDesc() : Dim(0,0,0,0) {
    }

    TensorDesc(int n, int c, int h, int w) : Dim(n,c,h,w) {
        CHECK_MIO(miopenCreateTensorDescriptor(&desc));
        CHECK_MIO(miopenSet4dTensorDescriptor(desc, miopenFloat, n, c, h, w));
    }
    TensorDesc(const Dim& dims) : Dim(dims) {
        CHECK_MIO(miopenCreateTensorDescriptor(&desc));
        CHECK_MIO(miopenSet4dTensorDescriptor(desc, miopenFloat, n, c, h, w));
    }

    TensorDesc(const TensorDesc& o) : TensorDesc(o.n, o.c, o.h, o.w) {}

    TensorDesc(TensorDesc&& o) {
        this->desc = o.desc;
        this->n = o.n;
        this->c = o.c;
        this->h = o.h;
        this->w = o.w;
        o.n = o.c = o.h = o.w = 0;
    }

    void free() {
        if (!(n == 0 && c == 0 && h == 0 && w == 0)) {
            CHECK_MIO(miopenDestroyTensorDescriptor(desc));
        }
    }

    ~TensorDesc() {
        free();
    }
};

std::ostream& operator<<(std::ostream& os, const TensorDesc& t) {
    return os << "(" << t.n << "," << t.c << "," << t.h << "," << t.w << ")";
}

struct Tensor : public TensorDesc {
    device_mem_t data;
    size_t data_size;
    Tensor() : TensorDesc(0,0,0,0) {
        data = NULL;
        data_size = 0;
    }

    Tensor(const Tensor& o) = default;
    Tensor(Tensor&& o) = default;

    Tensor(TensorDesc&& d) : TensorDesc(std::move(d)) {
        data_size = n;
        data_size *= c; data_size *= h; data_size *= w; data_size *= 4;
        DEBUG("Allocating Float Tensor (" << n << "," << c << "," << h << "," << h << "), total size: " << data_size / 1024 << " kB");
        data = device_alloc(data_size);
    }

    Tensor(const Dim& dims) : TensorDesc(dims) {
        data_size = n;
        data_size *= c; data_size *= h; data_size *= w; data_size *= 4;
        DEBUG("Allocating Float Tensor (" << n << "," << c << "," << h << "," << h << "), total size: " << data_size / 1024 << " kB");
        data = device_alloc(data_size);
    }

    Tensor(int n, int c, int h, int w) : TensorDesc(n, c, h, w) {
        data_size = n;
        data_size *= c; data_size *= h; data_size *= w; data_size *= 4;
        DEBUG("Allocating Float Tensor (" << n << "," << c << "," << h << "," << h << "), total size: " << data_size / 1024 << " kB");
        data = device_alloc(data_size);
    }

    ~Tensor() {
        device_free(data);
    }
};

struct ConvDesc {
    miopenConvolutionDescriptor_t desc;

    ConvDesc(int pad_h, int pad_w, int u, int v, int upscalex, int upscaley) {
        CHECK_MIO(miopenCreateConvolutionDescriptor(&desc));
        CHECK_MIO(miopenInitConvolutionDescriptor(desc, miopenConvolution, pad_h, pad_w, u, v, upscalex, upscaley));
    }

    // create with padding and stride, default upscale = 1
    ConvDesc(int pad_h, int pad_w, int u, int v) : ConvDesc(pad_h, pad_w, u, v, 1, 1) {
    }

    // default stride = 1, upscale = 1
    ConvDesc(int pad_h, int pad_w) : ConvDesc(pad_h, pad_w, 1, 1, 1, 1) {
    }

    // default pad = 0, stride = 1, upscale = 1
    ConvDesc() : ConvDesc(0, 0, 1, 1, 1, 1) {
    }

    ~ConvDesc() {
        CHECK_MIO(miopenDestroyConvolutionDescriptor(desc));
    }
};

struct ConvLayerDesc {
    int batch_size;
    int height;
    int width;
    int channels_in;
    int channels_out;
    int kernel_size;
};

struct Layer {
    TensorDesc input_desc;
    TensorDesc output_desc;

    Layer(const Dim& input_desc, const Dim& output_desc)
        : input_desc(input_desc), output_desc(output_desc) {}

    Layer(const TensorDesc& input_desc, const TensorDesc& output_desc)
        : input_desc(input_desc), output_desc(output_desc) {}

    // every layer has to implement forward and backward
    virtual void forward(const Tensor& input, Tensor& output) = 0;
    virtual void backward(const Tensor& doutput, Tensor& dinput) = 0;

    const TensorDesc& getInputDesc() const {
        return input_desc;
    }

    const TensorDesc& getOutputDesc() const {
        return output_desc;
    }
};

static Dim getConvOutputDim(int padding, int stride, const TensorDesc& input, const TensorDesc& weights) {
    int n, c, h, w;
    ConvDesc d(padding, padding, stride, stride, 1, 1);
    CHECK_MIO(miopenGetConvolutionForwardOutputDim(d.desc, input.desc, weights.desc, &n, &c, &h, &w));
    return Dim(n, c, h, w);
}

struct ConvLayer : public ConvDesc, public ConvLayerDesc, public Layer {
    Tensor weights;
    Tensor dweights;
    const Tensor* input_ref;

    DevBuffer buffer; // TODO: joined buffer for fwd/bwd

    // algorithm selection:
    miopenConvFwdAlgorithm_t fwd_algo;
    miopenConvBwdWeightsAlgorithm_t bwd_weights_algo;
    miopenConvBwdDataAlgorithm_t bwd_data_algo;


:
:redraw!
q
:
:w
    static Dim getOutputDim(const ConvDesc& convdesc, const TensorDesc& input, const TensorDesc& weights) {
        int n, c, h, w;
        CHECK_MIO(miopenGetConvolutionForwardOutputDim(convdesc.desc, input.desc, weights.desc, &n, &c, &h, &w));
        return Dim(n, c, h, w);
    }

    ConvLayer(const TensorDesc& input_dims, int channels_out, int kernel_size, int padding, int stride)
        : ConvDesc(padding, padding, stride, stride, 1, 1),
          ConvLayerDesc({input_dims.n, input_dims.h, input_dims.w, input_dims.c, channels_out, kernel_size}),
          Layer((Dim&)input_dims, getConvOutputDim(padding, stride, input_dims, TensorDesc(channels_out, input_dims.c, kernel_size, kernel_size))),
          weights(channels_out, input_dims.c, kernel_size, kernel_size),
          dweights(channels_out, input_dims.c, kernel_size, kernel_size)
    {
    }

    /* default stride = 1 */
    ConvLayer(const TensorDesc& input_dims, int channels_out, int kernel_size, int padding)
        : ConvLayer(input_dims, channels_out, kernel_size, padding, 1) {}

    /* default padding = 0, stride = 1 */
    ConvLayer(const TensorDesc& input_dims, int channels_out, int kernel_size)
        : ConvLayer(input_dims, channels_out, kernel_size, 0, 1) {}

    /* default padding = 0, stride = 1 */
    ConvLayer(const ConvLayerDesc& l)
        : ConvLayer(TensorDesc(l.batch_size, l.channels_in, l.height, l.width), l.channels_out, l.kernel_size) {}

    /*
    ConvLayer(int batch_size, int height, int width, int channels_in, int channels_out, int conv_size)
        : ConvDesc(), LayerDesc({batch_size, height, width, channels_in, channels_out, conv_size}),
          input(batch_size, channels_in, height, width),
          weights(channels_out, channels_in, conv_size, conv_size),
          //output(batch_size, channels_out, height - (conv_size-1)/2, width - (conv_size-1)/2)
          output(getOutputDim(*this, input, weights))
          //output(batch_size, channels_out, height, width)
    {
    }

    ConvLayer(const LayerDesc& l)
        : ConvDesc(),
          LayerDesc(l),
          input(batch_size, channels_in, height, width),
          weights(channels_out, channels_in, conv_size, conv_size),
          //output(batch_size, channels_out, height - (conv_size-1)/2, width - (conv_size-1)/2)
          output(getOutputDim(*this, input, weights))
          //output(batch_size, channels_out, height, width)
    {
    }
    */

    double num_flops() {
        return batch_size * 1.0 * height * width * channels_in * channels_out * kernel_size * kernel_size;
    }

    void init_forward(const Tensor& input, Tensor& output) {
        size_t workspace_size;
        CHECK_MIO(miopenConvolutionForwardGetWorkSpaceSize(mio::handle(), weights.desc, input.desc, this->desc, output.desc, &workspace_size));

        std::cout << "\tWorkspace size required for fwd: " << workspace_size << std::endl;
        if (workspace_size > buffer.size) {
            std::cout << "\tReallocating Buffer for larger workspace size" << std::endl;
            buffer = DevBuffer(workspace_size);
        }

        // find best algo, and benchmark!
        miopenConvAlgoPerf_t perfs[4];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionForwardAlgorithm(mio::handle(), input.desc, input.data, weights.desc, weights.data, this->desc, output.desc, output.data, 4, &returned_algos, perfs, buffer.data, buffer.size, false));

        std::cout << "\tMIOpen Found " << returned_algos << " fwd algorithms, choosing " << perfs[0].fwd_algo << ": " << std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            std::cout << "\t\t" << i << ") " << perfs[i].fwd_algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory << std::endl;
        }

        fwd_algo = perfs[0].fwd_algo;
        //fwd_algo = miopenConvolutionFwdAlgoDirect;
    }

    void find_bwd_data_algo(const Tensor& doutput, Tensor& dinput) {
        size_t workspace_size;
        CHECK_MIO(miopenConvolutionBackwardDataGetWorkSpaceSize(mio::handle(), doutput.desc, weights.desc, this->desc, dinput.desc, &workspace_size));

        std::cout << "\tWorkspace size required for bwd_data: " << workspace_size << std::endl;
        if (workspace_size > buffer.size) {
            std::cout << "\tReallocating Buffer for larger workspace size" << std::endl;
            buffer = DevBuffer(workspace_size);
        }

        // find best algo, and benchmark!
        miopenConvAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionBackwardDataAlgorithm(mio::handle(), doutput.desc, doutput.data, weights.desc, weights.data, this->desc, dinput.desc, dinput.data, 5, &returned_algos, perfs, buffer.data, buffer.size, false));

        std::cout << "\tMIOpen Found " << returned_algos << " bwd_data algorithms, choosing " << perfs[0].fwd_algo << ": " << std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            std::cout << "\t\t" << i << ") " << perfs[i].fwd_algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory << std::endl;
        }

        bwd_data_algo = perfs[0].bwd_data_algo;
        //bwd_data_algo = miopenConvolutionBwdDataAlgoDirect;
    }

    void find_bwd_weights_algo(const Tensor& doutput, Tensor& input) {
        size_t workspace_size;
        CHECK_MIO(miopenConvolutionBackwardWeightsGetWorkSpaceSize(mio::handle(), doutput.desc, input.desc, this->desc, weights.desc, &workspace_size));

        std::cout << "\tWorkspace size required for bwd_weights: " << workspace_size << std::endl;
        if (workspace_size > buffer.size) {
            std::cout << "\tReallocating Buffer for larger workspace size" << std::endl;
            buffer = DevBuffer(workspace_size);
        }

        // find best algo, and benchmark!
        miopenConvAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionBackwardWeightsAlgorithm(mio::handle(), doutput.desc, doutput.data, input.desc, input.data, this->desc, dweights.desc, dweights.data, 5, &returned_algos, perfs, buffer.data, buffer.size, false));

        std::cout << "\tMIOpen Found " << returned_algos << " bwd_weights algorithms, choosing " << perfs[0].fwd_algo << ": " << std::endl;
        for (int i = 0; i < returned_algos; ++i) {
            std::cout << "\t\t" << i << ") " << perfs[i].fwd_algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory << std::endl;
        }

        bwd_weights_algo = perfs[0].bwd_weights_algo;
        //bwd_weights_algo = miopenConvolutionBwdWeightsAlgoDirect;
    }

    /*
    void init(miopenHandle_t mio_handle) {
        init_fwd_algo(mio_handle);
        //find_bwd_data_algo(mio_handle);
        //find_bwd_weights_algo(mio_handle);
    }
    */
    void init_backward(const Tensor& doutput, Tensor& dinput) {
        find_bwd_data_algo(doutput, dinput);
        find_bwd_weights_algo(doutput, dinput);
    }

    void forward(const Tensor& input, Tensor& output) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenConvolutionForward(mio::handle(), &alpha, input.desc, input.data, weights.desc, weights.data, this->desc, fwd_algo, &beta, output.desc, output.data, buffer.data, buffer.size));
        // save for backward
        input_ref = &input;
    }


    void backward(const Tensor& doutput, Tensor& dinput) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenConvolutionBackwardData(mio::handle(), &alpha, doutput.desc, doutput.data, weights.desc, weights.data, this->desc, bwd_data_algo, &beta, dinput.desc, dinput.data, buffer.data, buffer.size));
        CHECK_MIO(miopenConvolutionBackwardWeights(mio::handle(), &alpha, doutput.desc, doutput.data, input_ref->desc, input_ref->data, this->desc, bwd_weights_algo, &beta, dweights.desc, dweights.data, buffer.data, buffer.size));
    }
};


struct MaxPool : public Layer {
    miopenPoolingDescriptor_t desc;

    // needed for backward: original input, original output, indeces (as workspace)
    DevBuffer indeces_buf;

    const Tensor* input;
    const Tensor* output;

    static Dim getOutputDim(const TensorDesc& input, int kernel_size, int padding, int stride) {
        int n, c, h, w;

        miopenPoolingDescriptor_t pool_desc;
        CHECK_MIO(miopenSet2dPoolingDescriptor(pool_desc, miopenPoolingMax, kernel_size, kernel_size, padding, padding, stride, stride));
        CHECK_MIO(miopenGetPoolingForwardOutputDim(pool_desc, input.desc, &n, &c, &h, &w));
        CHECK_MIO(miopenDestroyPoolingDescriptor(pool_desc));
        return Dim(n, c, h, w);
    }

    MaxPool(const TensorDesc& input_dim, int kernel_size, int padding, int stride) : Layer((Dim&)input_dim, getOutputDim(input_dim, kernel_size, padding, stride)) {
        CHECK_MIO(miopenSet2dPoolingDescriptor(desc, miopenPoolingMax, kernel_size, kernel_size, padding, padding, stride, stride));
    }

    ~MaxPool() {
        CHECK_MIO(miopenDestroyPoolingDescriptor(desc));
    }

    void init_forward() {
        size_t size;
        CHECK_MIO(miopenPoolingGetWorkSpaceSize(output_desc.desc, &size));
        indeces_buf = DevBuffer(size);
    }

    void forward(const Tensor& input, Tensor& output) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenPoolingForward(mio::handle(), desc, &alpha, input.desc, input.data, &beta, output.desc, output.data, true, indeces_buf.data, indeces_buf.size));
        // save for backward
        this->input = &input;
        this->output = &output;
    }

    void backward(const Tensor& doutput, Tensor& dinput) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenPoolingBackward(mio::handle(), desc, &alpha, getOutputDesc().desc, output->data, doutput.desc, doutput.data, getInputDesc().desc, input->data, &beta, dinput.desc, dinput.data, indeces_buf.data));
    }
};

struct ReLU : public Layer {
    miopenActivationDescriptor_t desc;

    const Tensor* input;
    const Tensor* output;

    /*
    ReLU() : Layer() {
        CHECK_MIO(miopenSetActivationDescriptor(desc, miopenActivationRELU, 0.0, 0.0, 0.0));
    }
    */

    ReLU(const TensorDesc& input_dim) : Layer(input_dim, input_dim) {
        CHECK_MIO(miopenSetActivationDescriptor(desc, miopenActivationRELU, 0.0, 0.0, 0.0));
    }


    ~ReLU() {
        CHECK_MIO(miopenDestroyActivationDescriptor(desc));
    }

    void forward(const Tensor& input, Tensor& output) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenActivationForward(mio::handle(), desc, &alpha, input.desc, input.data, &beta, output.desc, output.data));
        // save for backward
        this->input = &input;
        this->output = &output;
    }

    void backward(const Tensor& doutput, Tensor& dinput) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenActivationBackward(mio::handle(), desc, &alpha, output->desc, output->data, doutput.desc, doutput.data, input->desc, input->data, &beta, dinput.desc, dinput.data));
    }
};

void mm(const Tensor& A, bool transA, const Tensor& B, bool transB, Tensor& C) {
    assert(A.h == 1 && A.w == 1);
    assert(B.h == 1 && B.w == 1);
    assert(C.h == 1 && C.w == 1);
    int M = transA ? A.c : A.n;
    int K = transA ? A.n : A.c;
    assert(transB ? K == B.c : K == B.n);
    int N = transB ? B.n : B.c;
    assert(C.n == M && C.c == N); // result is MxN
    float alpha = 1.f;
    float beta = 0.f;
    // TODO: leading dimension lda, ldb, ldc?
    int lda = A.c;
    int ldb = B.c;
    int ldc = C.c;
    CHECK_MIO(miopenGemm(mio::handle(), false, transA, transB, M, N, K, &alpha, A.data, lda, B.data, ldb, &beta, C.data, ldc));
}

// (batch_size * size) -> (batch_size * size)
struct Linear {
    int batch_size;
    int in_size;
    int out_size;

    Tensor weights; // dim (out_channels, in_channels, 1, 1)
    Tensor dweights;

    const Tensor* input_ref;

    Linear(int batch_size, int in_size, int out_size)
        : batch_size(batch_size), in_size(in_size), out_size(out_size),
          weights(out_size, in_size, 1, 1), dweights(out_size, in_size, 1, 1)
    {
    }

    void forward(const Tensor& input, Tensor& output) {
        assert(batch_size == input.n);
        assert(batch_size == output.n);
        assert(out_size = output.c);
        assert(in_size == input.c);
        mm(input, false, weights, true, output); // O <- I * W^T
        input_ref = &input;
    }

    void backward(const Tensor& doutput, Tensor& dinput) {
        // two MMs
        mm(doutput, false, weights, false, dinput); // dI <- dO * W
        mm(doutput, true, *input_ref, false, dweights); // dW <- dO^T * I
    }
};

float getTemp() {
    std::ifstream f("/sys/class/hwmon/hwmon0/temp1_input");
    int temp;
    f >> temp;
    return temp / 1000.f; // temp is in milli celsius
}

int getFanspeed() {
    std::ifstream f("/sys/class/hwmon/hwmon0/pwm1");
    int fan;
    f >> fan;
    return fan;
}

int getClock() {
    std::ifstream f("/sys/class/drm/card0/device/pp_dpm_sclk");
    std::string line;
    while (std::getline(f, line)) {
        if (line.back() == '*') {
            std::string mhzstr = line.substr(3, line.size()-3-5);
            std::istringstream iss(mhzstr);
            int mhz;
            iss >> mhz;
            return mhz;
        }
    }
    return 0;
}

#if 0
void alex() {
    Dim input_dim(batch_size, 3, x, y);
    /* convolutions: input_dims, output_channels, kernel_size, padding, stride */
    ConvLayer c1(input_dim, 64, 11, 2, 4);
    // ReLU
    ReLU r1();
    // MaxPool2D kernel_size = 3, stride = 2
    MaxPool max1();
    ConvLayer c2(prev.output_dims(), 64, 192, 5, 2, 0);
}
#endif


int main(int argc, char *argv[])
{
    device_init();

    // enable profiling
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    // batch_size, w, h, channels_in, channels_out, filter_size (eg, 3 for 3x3)
    /*
    std::vector<ConvLayerDesc> runs = {{128, 13, 13, 384, 384, 3},
                                   {128, 16, 16, 128, 128, 7},
                                   {128, 32, 32, 128, 128, 9},
                                   {128, 64, 64, 64, 128, 9},
                                   {128, 128, 128, 3, 96, 11}};
                                   */


    std::vector<ConvLayerDesc> runs = {{128, 64, 64, 64, 128, 3}};
                                   //{64, 32, 32, 18, 18, 3}};
                                   //{64, 32, 32, 16, 16, 5}};


    int layer = 5;
    int reps = 30;
    //LayerDesc& l = runs[3];
    for (ConvLayerDesc& l : runs) {

        TensorDesc input_dim(128, 64, 64, 64);
        ConvLayer conv(input_dim, 128, 3, 0, 1);

        Tensor input(input_dim);
        Tensor output(conv.getOutputDesc());


        std::ofstream of("benchmark.log");
        of << "Time\tTemp\tFan\tClock" << std::endl;
        of << 0 << "\t" << getTemp() << "\t" << getFanspeed() << "\t" << getClock() << std::endl;
        conv.init_forward(input, output);
        //conv.init_backward(output, input);
        of << 0 << "\t" << getTemp() << "\t" << getFanspeed() << "\t" << getClock() << std::endl;


        {
            CHECK_HIP(hipDeviceSynchronize());
            auto tic = std::chrono::steady_clock::now();
            std::vector<float> conv_times(reps);
            for (int i = 0; i < reps; ++i) {
                conv.forward(input, output);
                int clk = getClock();
                int fan = getFanspeed();
                float temp = getTemp();
                CHECK_MIO(miopenGetKernelTime(mio::handle(), &conv_times[i]));
                of << conv_times[i] << "\t" << temp << "\t" << fan << "\t" << clk << std::endl;
                std::cout << conv_times[i] << "\t" << temp << "\t" << fan << "\t" << clk << std::endl;
            }
            CHECK_HIP(hipDeviceSynchronize());
            auto toc = std::chrono::steady_clock::now();
            double time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count()*1.0/reps;
            double mflop = conv.num_flops() / 1024.0 / 1024.0;
            std::cout << "theo mflop: " << mflop << std::endl;
            std::cout << "Time for FWD L" << layer << ": " << time << " ms, " << mflop/time << " GFlops" << std::endl;
            //std::cout << "    Time per launch: ";
            //for (int i = 0; i < reps; ++i) {
            //    std::cout << conv_times[i] << ", ";
            //}
            //std::cout << std::endl;
        }
        /*
        {
            CHECK_HIP(hipDeviceSynchronize());
            auto tic = std::chrono::steady_clock::now();
            std::vector<float> conv_times(reps);
            for (int i = 0; i < reps; ++i) {
                conv.backward_data(mio_handle);
                //CHECK_MIO(miopenGetKernelTime(mio_handle, &conv_times[i]));
                conv.backward_weights(mio_handle);
                //CHECK_MIO(miopenGetKernelTime(mio_handle, &conv_times[i]));
                //std::cout << conv_times[i] << "\t" << getTemp() << "\t" << getFanspeed() << "\t" << getClock() << std::endl;
                //of << conv_times[i] << "\t" << getTemp() << "\t" << getFanspeed() << "\t" << getClock() << std::endl;
            }
            CHECK_HIP(hipDeviceSynchronize());
            auto toc = std::chrono::steady_clock::now();
            double time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count()*1.0/reps;
            //double mflop = conv.num_flops() / 1024.0 / 1024.0;
            std::cout << "Time for BWD L" << layer << ": " << time << " ms" << std::endl;
            std::cout << "    Time per launch: ";
            for (int i = 0; i < reps; ++i) {
                std::cout << conv_times[i] << ", ";
            }
            std::cout << std::endl;
        }
        */
        --layer;
    }


    miopenDestroy(mio::handle());
    return 0;
}

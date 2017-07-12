
#include <assert.h>

#include <iostream>
#include <chrono>
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>


#include <miopen/miopen.h>
#include <hipblas.h>
#include <gperftools/profiler.h>

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

#define DEBUG(msg) std::cerr << "[DEBUG] " << msg << std::endl;
#define INFO(msg) std::cerr << "[INFO]  " << msg << std::endl;

#define CHECK_MIO(cmd) \
{\
    miopenStatus_t miostat = cmd;\
    if (miostat != miopenStatusSuccess) { \
        fprintf(stderr, "error: '%s'(%d) at %s:%d\n", mio_err[(int)miostat], miostat,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}

// get miopenHandle globally via `mio::handle()`
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

// timer that prints out its name and time elapsed between construction and
// destruction
struct timed_section {
    std::string name;
    std::chrono::steady_clock::time_point tic;
    timed_section(const std::string& name) : name(name) {
        tic = std::chrono::steady_clock::now();
    }

    ~timed_section() {
        auto toc = std::chrono::steady_clock::now();
        double time = std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() / 1000.0;
        INFO("Section `" << name << "`\ttime: " << time << " ms");
    }
};

// class for wrapping around device buffers
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

    TensorDesc& operator=(TensorDesc&& o) {
        this->desc = o.desc;
        this->n = o.n;
        this->c = o.c;
        this->h = o.h;
        this->w = o.w;
        o.n = o.c = o.h = o.w = 0;
        return *this;
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
    bool owns_data;
    Tensor() : TensorDesc(0,0,0,0), owns_data(false) {
        data = NULL;
        data_size = 0;
    }

    //Tensor(const Tensor& o) = default;
    Tensor(Tensor&& o)
        : TensorDesc(std::move(o)),
          owns_data(o.owns_data),
          data(o.data),
          data_size(o.data_size)
    {
        o.data = nullptr;
        o.data_size = 0;
        o.owns_data = false;
    }

    Tensor& operator=(Tensor&& o) {
        TensorDesc::operator=(std::move(o));
        this->owns_data = o.owns_data;
        this->data = o.data;
        this->data_size = o.data_size;
        o.data = nullptr;
        o.data_size = 0;
        o.owns_data = false;
        return *this;
    }

    std::vector<float> toHost() {
        std::vector<float> x(data_size/sizeof(float));
        hipMemcpyDtoH(&x[0], data, data_size);
        return x;
    }

    void fromHost(const std::vector<float>& h) {
        hipMemcpyHtoD(data,(void*) h.data(), data_size);
        hipDeviceSynchronize();
    }

    void alloc() {
        DEBUG("Allocating Float Tensor (" << n << "," << c << "," << h << "," << h << "), total size: " << data_size / 1024 << " kB");
        data = device_alloc(data_size);
    }

    // randomly initiate tensor via copying from host
    void uniform() {
        std::vector<float> h(data_size/sizeof(float));
        std::generate(h.begin(), h.end(), [](){return rand()*1.f/RAND_MAX;});
        hipMemcpyHtoD(data, h.data(), data_size);
    }


    Tensor(TensorDesc&& d)
        : TensorDesc(std::move(d)),
          owns_data(true),
          data_size(n*(size_t)c*h*w*sizeof(float)) {
        alloc();
    }

    Tensor(const Dim& dims)
        : TensorDesc(dims),
          owns_data(true),
          data_size(n*(size_t)c*h*w*sizeof(float)) {
        alloc();
    }

    Tensor(int n, int c, int h, int w)
        : TensorDesc(n, c, h, w),
          owns_data(true),
          data_size(n*(size_t)c*h*w*sizeof(float)) {
        alloc();
    }

    Tensor(int n, int c, int h, int w, bool do_alloc)
        : TensorDesc(n, c, h, w),
          owns_data(do_alloc),
          data_size(n*(size_t)c*h*w*sizeof(float)) {
        if (do_alloc) {
            alloc();
        }
    }

    // reshape (creates a tensor object of new dimensions that doesn't own its data)
    Tensor viewAs(int n, int c, int h, int w) const {
        Tensor t(n, c, h, w, false);
        assert(n == this->n);
        assert(c*h*w == this->c * this->h * this->w);
        t.data = this->data;
        t.data_size = this->data_size;
        return t;
    }

    Tensor viewAs(const TensorDesc& d) const {
        return viewAs(d.n, d.c, d.h, d.w);
    }

    ~Tensor() {
        if (owns_data && data_size > 0) {
            device_free(data);
        }
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

// parameters for a 2D convolutional layer
struct ConvLayerDesc {
    int batch_size;
    int height;
    int width;
    int channels_in;
    int channels_out;
    int kernel_size;
    int padding;
    int stride;
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
    virtual void init_forward(const Tensor& input, Tensor& output) {};
    virtual void backward(const Tensor& doutput, Tensor& dinput) = 0;
    virtual void init_backward(const Tensor& doutput, Tensor& dinput) {};

    const TensorDesc& getInputDesc() const {
        return input_desc;
    }

    const TensorDesc& getOutputDesc() const {
        return output_desc;
    }

    std::ostream& write_dims(std::ostream& os) const {
        return os << getInputDesc() << " -> " << getOutputDesc();
    }

    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "Layer (unknown)";
    }

    virtual std::ostream& write(std::ostream& os) const {
        return this->write_dims(this->write_name(os) << ":\t");
    }
};

std::ostream& operator<<(std::ostream& os, const Layer& l) {
    return l.write(os);
}

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


    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "Conv(" << kernel_size << "x" << kernel_size << ")\t";
    }

    ConvLayer(const TensorDesc& input_dims, int channels_out, int kernel_size, int padding, int stride)
        : ConvDesc(padding, padding, stride, stride, 1, 1),
          ConvLayerDesc({input_dims.n, input_dims.h, input_dims.w, input_dims.c, channels_out, kernel_size, padding, stride}),
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

    /* construct via conv parameters */
    ConvLayer(const ConvLayerDesc& l)
        : ConvLayer(TensorDesc(l.batch_size, l.channels_in, l.height, l.width), l.channels_out, l.kernel_size, l.padding, l.stride) {}

    // estimate the number of muliplications for a direct implementation
    double num_flops() {
        return batch_size * 1.0 * height * width * channels_in * channels_out * kernel_size * kernel_size;
    }

    void init_forward(const Tensor& input, Tensor& output) override {
        size_t workspace_size;
        CHECK_MIO(miopenConvolutionForwardGetWorkSpaceSize(mio::handle(), weights.desc, input.desc, this->desc, output.desc, &workspace_size));

        //std::cout << "\tWorkspace size required for fwd: " << workspace_size << std::endl;
        if (workspace_size > buffer.size) {
            buffer = DevBuffer(workspace_size);
        }

        // find best algo, and benchmark!
        miopenConvAlgoPerf_t perfs[4];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionForwardAlgorithm(mio::handle(), input.desc, input.data, weights.desc, weights.data, this->desc, output.desc, output.data, 4, &returned_algos, perfs, buffer.data, buffer.size, false));

        INFO("\tMIOpen Found " << returned_algos << " fwd algorithms, choosing " << perfs[0].fwd_algo << ": ");
        for (int i = 0; i < returned_algos; ++i) {
            INFO("\t\t" << i << ") " << perfs[i].fwd_algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory);
        }

        fwd_algo = perfs[0].fwd_algo;
    }

    void find_bwd_data_algo(const Tensor& doutput, Tensor& dinput) {
        size_t workspace_size;
        CHECK_MIO(miopenConvolutionBackwardDataGetWorkSpaceSize(mio::handle(), doutput.desc, weights.desc, this->desc, dinput.desc, &workspace_size));

        //std::cout << "\tWorkspace size required for bwd_data: " << workspace_size << std::endl;
        if (workspace_size > buffer.size) {
            buffer = DevBuffer(workspace_size);
        }

        // find best algo, and benchmark!
        miopenConvAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionBackwardDataAlgorithm(mio::handle(), doutput.desc, doutput.data, weights.desc, weights.data, this->desc, dinput.desc, dinput.data, 5, &returned_algos, perfs, buffer.data, buffer.size, false));

        INFO("\tMIOpen Found " << returned_algos << " bwd_data algorithms, choosing " << perfs[0].fwd_algo << ": ");
        for (int i = 0; i < returned_algos; ++i) {
            INFO("\t\t" << i << ") " << perfs[i].fwd_algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory);
        }

        bwd_data_algo = perfs[0].bwd_data_algo;
    }

    void find_bwd_weights_algo(const Tensor& doutput, Tensor& input) {
        size_t workspace_size;
        CHECK_MIO(miopenConvolutionBackwardWeightsGetWorkSpaceSize(mio::handle(), doutput.desc, input.desc, this->desc, weights.desc, &workspace_size));

        //std::cout << "\tWorkspace size required for bwd_weights: " << workspace_size << std::endl;
        if (workspace_size > buffer.size) {
            buffer = DevBuffer(workspace_size);
        }

        // find best algo, and benchmark!
        miopenConvAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionBackwardWeightsAlgorithm(mio::handle(), doutput.desc, doutput.data, input.desc, input.data, this->desc, dweights.desc, dweights.data, 5, &returned_algos, perfs, buffer.data, buffer.size, false));

        INFO("\tMIOpen Found " << returned_algos << " bwd_weights algorithms, choosing " << perfs[0].fwd_algo << ": ");
        for (int i = 0; i < returned_algos; ++i) {
            INFO("\t\t" << i << ") " << perfs[i].fwd_algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory);
        }

        bwd_weights_algo = perfs[0].bwd_weights_algo;
    }

    void init_backward(const Tensor& doutput, Tensor& dinput) override {
        find_bwd_data_algo(doutput, dinput);
        find_bwd_weights_algo(doutput, dinput);
    }

    void forward(const Tensor& input, Tensor& output) override {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenConvolutionForward(mio::handle(), &alpha, input.desc, input.data, weights.desc, weights.data, this->desc, fwd_algo, &beta, output.desc, output.data, buffer.data, buffer.size));
        // save for backward
        input_ref = &input;
    }

    void backward(const Tensor& doutput, Tensor& dinput) override {
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

    int kernel_size, padding, stride;

    static Dim getOutputDim(const TensorDesc& input, int kernel_size, int padding, int stride) {
        int n, c, h, w;

        miopenPoolingDescriptor_t pool_desc;
        CHECK_MIO(miopenCreatePoolingDescriptor(&pool_desc));
        CHECK_MIO(miopenSet2dPoolingDescriptor(pool_desc, miopenPoolingMax, kernel_size, kernel_size, padding, padding, stride, stride));
        CHECK_MIO(miopenGetPoolingForwardOutputDim(pool_desc, input.desc, &n, &c, &h, &w));
        CHECK_MIO(miopenDestroyPoolingDescriptor(pool_desc));
        return Dim(n, c, h, w);
    }

    virtual std::ostream& write_name(std::ostream& os) const {
        //return os << "MaxPool(" << kernel_size << "x" << kernel_size << ", pad=" << padding << ",stride=" << stride << ")";
        return os << "MaxPool(" << kernel_size << "x" << kernel_size << ")";
    }

    MaxPool(const TensorDesc& input_dim, int kernel_size, int padding, int stride)
        : Layer((Dim&)input_dim, MaxPool::getOutputDim(input_dim, kernel_size, padding, stride)),
          kernel_size(kernel_size), padding(padding), stride(stride) {
        CHECK_MIO(miopenCreatePoolingDescriptor(&desc));
        CHECK_MIO(miopenSet2dPoolingDescriptor(desc, miopenPoolingMax, kernel_size, kernel_size, padding, padding, stride, stride));
    }

    ~MaxPool() {
        CHECK_MIO(miopenDestroyPoolingDescriptor(desc));
    }

    void init_forward(const Tensor& input, Tensor& output) {
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

    const Tensor* input_ref;
    const Tensor* output_ref;


    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "ReLU()\t";
    }

    ReLU(const TensorDesc& input_dim) : Layer(input_dim, input_dim) {
        CHECK_MIO(miopenCreateActivationDescriptor(&desc));
        CHECK_MIO(miopenSetActivationDescriptor(desc, miopenActivationRELU, 0.0, 0.0, 1.0));
    }


    ~ReLU() {
        CHECK_MIO(miopenDestroyActivationDescriptor(desc));
    }

    void forward(const Tensor& input, Tensor& output) {
        /*
        float alpha = 1.f;
        float beta = 0.f;
        */
        int alpha = 1, beta = 1;
        CHECK_MIO(miopenActivationForward(mio::handle(), desc, &alpha, input.desc, input.data, &beta, output.desc, output.data));
        // save for backward
        this->input_ref = &input;
        this->output_ref = &output;
    }

    void backward(const Tensor& doutput, Tensor& dinput) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenActivationBackward(mio::handle(), desc, &alpha, output_ref->desc, output_ref->data, doutput.desc, doutput.data, input_ref->desc, input_ref->data, &beta, dinput.desc, dinput.data));
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
    //int lda = transA ? M : K;
    int lda = A.c;
    //int ldb = transB ? K : N;
    int ldb = B.c;
    int ldc = N;
    assert(A.data_size == M*K*4);
    assert(B.data_size == K*N*4);
    assert(C.data_size == M*N*4);
    CHECK_MIO(miopenGemm(mio::handle(), false, transA, transB, M, N, K, &alpha, A.data, lda, B.data, ldb, &beta, C.data, ldc));
}

void mm_blas(const Tensor& A, bool transA, const Tensor& B, bool transB, Tensor& C) {
    assert(A.h == 1 && A.w == 1);
    assert(B.h == 1 && B.w == 1);
    assert(C.h == 1 && C.w == 1);

    int M = transA ? A.c : A.n;
    int K = transA ? A.n : A.c;
    int N = transB ? B.n : B.c;
    assert(transB ? K == B.c : K == B.n);
    assert(C.n == M && C.c == N);

    float alpha = 1.f;
    float beta = 0.f;
    int lda = A.c;
    int ldb = B.c;
    int ldc = C.c;
    hipblasHandle_t blas_handle;
    hipblasCreate(&blas_handle);
    hipblasOperation_t opA = transA ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    hipblasOperation_t opB = transB ? HIPBLAS_OP_T : HIPBLAS_OP_N;
    // call Sgemm with A<->B swapped (since we have rowmaj, but blas expects colmajor)
    hipblasStatus_t err = hipblasSgemm(blas_handle, opB, opA, N, M, K, &alpha, (const float*)B.data, ldb, (const float*)A.data, lda, &beta, (float*)C.data, ldc);
    assert(err == 0);
}

// (batch_size * size) -> (batch_size * size)
struct Linear : public Layer {
    int batch_size;
    int in_size;
    int out_size;

    Tensor weights; // dim (out_channels, in_channels, 1, 1)
    Tensor dweights;

    const Tensor* input_ref;

    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "Linear(" << in_size << "," << out_size << ")";
    }

    Linear(const TensorDesc& input_dim, int out_size)
        : Layer(input_dim, TensorDesc(input_dim.n, out_size, 1, 1)),
          batch_size(input_dim.n),
          in_size(input_dim.c * input_dim.h * input_dim.w),
          out_size(out_size),
          weights(out_size, in_size, 1, 1),
          dweights(out_size, in_size, 1, 1)
    {
    }

    void forward(const Tensor& input, Tensor& output) {
        assert(batch_size == input.n);
        assert(batch_size == output.n);
        assert(out_size = output.c);
        assert(in_size == input.c * input.h * input.w);
        mm_blas(input, false, weights, true, output); // O <- I * W^T
        input_ref = &input;
    }

    void backward(const Tensor& doutput, Tensor& dinput) {
        // two MMs
        mm_blas(doutput, true, *input_ref, false, dweights); // dW <- dO^T * I
        mm_blas(doutput, false, weights, false, dinput); // dI <- dO * W
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

struct Reshape : public Layer {

    Reshape(const TensorDesc& input_dim, int n, int c, int h, int w)
        : Layer(input_dim, TensorDesc(n, c, h, w)) {
        assert(input_dim.n == n);
        assert(input_dim.c * input_dim.h * input_dim.w == c*h*w);
    }


    void init_forward(const Tensor& input, Tensor& output) override {
        output = std::move(input.viewAs(getOutputDesc()));
    }

    void forward(const Tensor& input, Tensor& output) override {
        output = std::move(input.viewAs(getOutputDesc()));
    }

    void init_backward(const Tensor& doutput, Tensor& dinput) override {
        dinput = std::move(doutput.viewAs(getInputDesc()));
    }

    void backward(const Tensor& doutput, Tensor& dinput) override {
        dinput = std::move(doutput.viewAs(getInputDesc()));
    }
};

struct Model {
    Tensor input;
    bool is_init_fwd;
    bool is_init_bwd;
    std::vector<std::shared_ptr<Layer>> layers;
    std::vector<std::shared_ptr<Tensor>> out_tensors;

    Model(const Dim& input_dim) : input(input_dim) {}

    const TensorDesc& last_output_dim() {
        if (layers.empty()) {
            return input;
        } else {
            return layers.back()->getOutputDesc();
        }
    }

    void addConv(int output_channels, int kernel_size, int padding, int stride) {
        layers.emplace_back(new ConvLayer(last_output_dim(), output_channels, kernel_size, padding, stride));
        out_tensors.emplace_back(new Tensor(layers.back()->getOutputDesc()));
    }

    void addReLU() {
        layers.emplace_back(new ReLU(last_output_dim()));
        // TODO: enable inplace ReLU
        out_tensors.emplace_back(new Tensor(layers.back()->getOutputDesc()));
    }

    void addMaxPool(int kernel_size, int padding, int stride) {
        layers.emplace_back(new MaxPool(last_output_dim(), kernel_size, padding, stride));
        out_tensors.emplace_back(new Tensor(layers.back()->getOutputDesc()));
    }

    void addLinear(int outsize) {
        layers.emplace_back(new Linear(last_output_dim(), outsize));
        out_tensors.emplace_back(new Tensor(layers.back()->getOutputDesc()));
    }

    void reshape(int n, int c, int h, int w) {
        layers.emplace_back(new Reshape(last_output_dim(), n, c, h, w));
        out_tensors.emplace_back(new Tensor(n, c, h, w, false)); /* Tensor data gets set in forward() */
    }

    // for each layer, calls f(Layer& l, Tensor& in, Tensor& out);
    template <typename Func>
    void forward_pass(Func f) {
        Tensor* in = &input;
        Tensor* out;

        for (size_t i = 0; i < layers.size(); ++i) {
            out = out_tensors[i].get();
            f(*layers[i], *in, *out);
            in = out;
        }
    }

    // for each layer backwards, calls b(Layer& l, Tensor& dout, Tensor& din)
    template <typename Func>
    void backward_pass(Func b) {
        assert(out_tensors.size() > 0);
        Tensor* dout = out_tensors.back().get();
        Tensor* din;
        for (size_t i = 0; i < layers.size(); ++i) {
            if (i < layers.size()-1) {
                din = out_tensors[out_tensors.size()-i-2].get();
            } else {
                din = &input;
            }
            b(*layers[out_tensors.size()-i-1], *dout, *din);
            dout = din;
        }
    }


    // initializes all layers for fwd
    void init_forward() {
        forward_pass([](Layer& l, Tensor& in, Tensor& out){l.init_forward(in, out);});
        is_init_fwd = true;
    }

    void forward() {
        if (!is_init_fwd) {
            init_forward();
        }
        forward_pass([](Layer& l, Tensor& in, Tensor& out){
            std::stringstream ss;
            ss << "Fwd " << l;
            timed_section s(ss.str());
            l.forward(in, out);
            CHECK_HIP(hipDeviceSynchronize());
        });
    }

    void init_backward() {
        backward_pass([](Layer& l, Tensor& dout, Tensor& din){l.init_backward(dout, din);});
        is_init_bwd = true;
    }

    void backward() {
        if (!is_init_bwd) {
            init_backward();
        }
        backward_pass([](Layer& l, Tensor& dout, Tensor& din) {
            std::stringstream ss;
            ss << "Bwd " << l;
            timed_section s(ss.str());
            l.backward(dout, din);
            CHECK_HIP(hipDeviceSynchronize());
        });
    }
};



void benchmark_convlayers() {
    // batch_size, w, h, channels_in, channels_out, kernel_size, padding, stride
    std::vector<ConvLayerDesc> runs = {{128, 13, 13, 384, 384, 3, 0, 1},
                                   {128, 16, 16, 128, 128, 7, 0, 1},
                                   {128, 32, 32, 128, 128, 9, 0, 1},
                                   {128, 64, 64, 64, 128, 9, 0, 1},
                                   {128, 128, 128, 3, 96, 11, 0, 1}};


    /*
    std::vector<ConvLayerDesc> runs = {{128, 64, 64, 64, 128, 3, 0, 1},
                                       {128, 64, 64, 64, 128, 3, 1, 1},
                                       {128, 28, 28, 64, 64, 5, 1, 2}};
                                       */


    int layer = 5;
    int reps = 10;
    //LayerDesc& l = runs[3];
    for (ConvLayerDesc& l : runs) {

        //TensorDesc input_dim(128, 64, 64, 64);
        //ConvLayer conv(input_dim, 128, 3, 0, 1);
        ConvLayer conv(l);

        Tensor input(conv.getInputDesc());
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
            std::cout << "Time for FWD L" << layer << ": " << time << " ms" << std::endl;
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

}

/* TODO:
 * - [ ] create AlexNet class
 * - [ ] uniform random tensors (via host->device copy), and CPU initialized tensors
 * - [ ] simple print tensor function for verifying mm_blas
 * - [ ] Make `Model` take input and output tensors in forward(), backward()
 * - [ ] Create runner and benchmarker class
 */

void alexNet() {
    int reps = 10;
    TensorDesc input_dim(128, 3, 224, 224);

    Model m(input_dim);
    /* features */
    m.addConv(64, 11, 2, 4);
    m.addReLU();
    m.addMaxPool(3, 0, 2);
    m.addConv(192, 5, 2, 1);
    m.addReLU();
    m.addMaxPool(3, 0, 2);
    m.addConv(384, 3, 1, 1);
    m.addReLU();
    m.addConv(256, 3, 1, 1);
    m.addReLU();
    m.addConv(256, 3, 1, 1);
    m.addReLU();
    m.addMaxPool(3, 0, 2);

    DEBUG("DIms after Features; " << m.last_output_dim());

    /* classifier */
    // TODO Dropout
    m.reshape(128, 256 * 6 * 6, 1, 1);
    m.addLinear(4096);
    m.addReLU();
    // TODO: Dropout
    m.addLinear(4096);
    m.addReLU();
    m.addLinear(1000);

    INFO("Init fwd");
    m.init_forward();
    INFO("Init bwd");
    m.init_backward();

    INFO("Begin warmup runs");
    for (int i = 0; i < 3; ++i) {
        {
            INFO("               ======= BEGIN FWD =======");
            timed_section s("Fwd Pass");
            m.forward();
            CHECK_HIP(hipDeviceSynchronize());
        }
        {
            INFO("               ======= BEGIN BWD =======");
            timed_section s("Bwd Pass");
            m.backward();
            CHECK_HIP(hipDeviceSynchronize());
        }
    }

    INFO("Begin Timings");
    std::chrono::steady_clock::duration fwdtime;
    auto tic = std::chrono::steady_clock::now();
    for (int i = 0; i < reps; ++i) {
        {
            INFO("               ======= BEGIN FWD =======");
            timed_section s("Fwd Pass");
            //auto tic = std::chrono::steady_clock::now();
            m.forward();
            //auto toc = 
        }
        {
            INFO("               ======= BEGIN BWD =======");
            timed_section s("Bwd Pass");
            m.backward();
        }
    }
    auto toc = std::chrono::steady_clock::now();
    double time = std::chrono::duration_cast<std::chrono::milliseconds>(toc - tic).count()*1.0/reps;
    INFO("Avg time per fwd+bwd: " << time << " ms");
}

int main(int argc, char *argv[])
{
    device_init();

    // enable profiling
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    alexNet();
    //benchmark_convlayers();

    miopenDestroy(mio::handle());
    return 0;
}

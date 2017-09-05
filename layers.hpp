#ifndef LAYERS_HPP
#define LAYERS_HPP

#include "tensor.hpp"
#include "function.hpp"

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

    // algorithm selection:
    miopenConvFwdAlgorithm_t fwd_algo;
    miopenConvBwdWeightsAlgorithm_t bwd_weights_algo;
    miopenConvBwdDataAlgorithm_t bwd_data_algo;


    virtual std::ostream& write_name(std::ostream& os) const {
        //return os << "Conv(" << kernel_size << "x" << kernel_size << ")";
        return os << "Conv(" << kernel_size << "x" << kernel_size << ",pad=" << padding << ",s=" << stride << ")";
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
        size_t fwd_workspace_size;
        CHECK_MIO(miopenConvolutionForwardGetWorkSpaceSize(mio::handle(), weights.desc, input.desc, this->desc, output.desc, &fwd_workspace_size));
        DEBUG("Init fwd " << *this << " req workspace: " << fwd_workspace_size);

        DevBuffer& buffer = WorkSpace::get(fwd_workspace_size);

        // find best algo, and benchmark!
        miopenConvAlgoPerf_t perfs[4];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionForwardAlgorithm(mio::handle(), input.desc, input.data, weights.desc, weights.data, this->desc, output.desc, output.data, 4, &returned_algos, perfs, buffer.data, fwd_workspace_size, false));

        INFO("\tMIOpen Found " << returned_algos << " fwd algorithms, choosing " << perfs[0].fwd_algo << ": ");
        for (int i = 0; i < returned_algos; ++i) {
            INFO("\t\t" << i << ") " << perfs[i].fwd_algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory);
        }

        fwd_algo = perfs[0].fwd_algo;
    }

    void find_bwd_data_algo(const Tensor& doutput, Tensor& dinput) {
        size_t bwd_data_workspace_size;
        CHECK_MIO(miopenConvolutionBackwardDataGetWorkSpaceSize(mio::handle(), doutput.desc, weights.desc, this->desc, dinput.desc, &bwd_data_workspace_size));
        DEBUG("Init bwd_data " << *this << " req workspace: " << bwd_data_workspace_size);

        DevBuffer& buffer = WorkSpace::get(bwd_data_workspace_size);

        // find best algo, and benchmark!
        miopenConvAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionBackwardDataAlgorithm(mio::handle(), doutput.desc, doutput.data, weights.desc, weights.data, this->desc, dinput.desc, dinput.data, 5, &returned_algos, perfs, buffer.data, bwd_data_workspace_size, false));

        INFO("\tMIOpen Found " << returned_algos << " bwd_data algorithms, choosing " << perfs[0].fwd_algo << ": ");
        for (int i = 0; i < returned_algos; ++i) {
            INFO("\t\t" << i << ") " << perfs[i].fwd_algo << " - time: " << perfs[i].time << ", Memory: " << perfs[i].memory);
        }

        bwd_data_algo = perfs[0].bwd_data_algo;
    }

    void find_bwd_weights_algo(const Tensor& doutput, Tensor& input) {
        size_t bwd_weights_workspace_size;
        CHECK_MIO(miopenConvolutionBackwardWeightsGetWorkSpaceSize(mio::handle(), doutput.desc, input.desc, this->desc, weights.desc, &bwd_weights_workspace_size));
        DEBUG("Init bwd_weights " << *this << " req workspace: " << bwd_weights_workspace_size);

        DevBuffer& buffer = WorkSpace::get(bwd_weights_workspace_size);

        // find best algo, and benchmark!
        miopenConvAlgoPerf_t perfs[5];
        int returned_algos;
        CHECK_MIO(miopenFindConvolutionBackwardWeightsAlgorithm(mio::handle(), doutput.desc, doutput.data, input.desc, input.data, this->desc, dweights.desc, dweights.data, 5, &returned_algos, perfs, buffer.data, bwd_weights_workspace_size, false));

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
        DevBuffer& buffer = WorkSpace::get();
        CHECK_MIO(miopenConvolutionForward(mio::handle(), &alpha, input.desc, input.data, weights.desc, weights.data, this->desc, fwd_algo, &beta, output.desc, output.data, buffer.data, buffer.size));
        // save for backward
        input_ref = &input;
    }

    void backward(const Tensor& doutput, Tensor& dinput) override {
        float alpha = 1.f;
        float beta = 0.f;
        DevBuffer& buffer = WorkSpace::get();
        CHECK_MIO(miopenConvolutionBackwardData(mio::handle(), &alpha, doutput.desc, doutput.data, weights.desc, weights.data, this->desc, bwd_data_algo, &beta, dinput.desc, dinput.data, buffer.data, buffer.size));
        CHECK_MIO(miopenConvolutionBackwardWeights(mio::handle(), &alpha, doutput.desc, doutput.data, input_ref->desc, input_ref->data, this->desc, bwd_weights_algo, &beta, dweights.desc, dweights.data, buffer.data, buffer.size));
    }
};


struct PoolingLayer : public Layer {
    miopenPoolingMode_t pool_mode;
    miopenPoolingDescriptor_t desc;

    // needed for backward: original input, original output, indeces (as workspace)
    DevBuffer indeces_buf;

    const Tensor* input;
    const Tensor* output;

    int kernel_size, padding, stride;

    static Dim getOutputDim(const TensorDesc& input, int kernel_size, int padding, int stride, miopenPoolingMode_t pool_mode) {
        int n, c, h, w;

        miopenPoolingDescriptor_t pool_desc;
        CHECK_MIO(miopenCreatePoolingDescriptor(&pool_desc));
        CHECK_MIO(miopenSet2dPoolingDescriptor(pool_desc, pool_mode, kernel_size, kernel_size, padding, padding, stride, stride));
        CHECK_MIO(miopenGetPoolingForwardOutputDim(pool_desc, input.desc, &n, &c, &h, &w));
        CHECK_MIO(miopenDestroyPoolingDescriptor(pool_desc));
        return Dim(n, c, h, w);
    }

    virtual std::ostream& write_name(std::ostream& os) const override {
        if (pool_mode == miopenPoolingMax)
            os << "MaxPool(";
        else
            os << "AvgPool(";
        return os << kernel_size << "x" << kernel_size << ")";
    }

    PoolingLayer(const TensorDesc& input_dim, int kernel_size, int padding, int stride, miopenPoolingMode_t pool_mode)
        : Layer((Dim&)input_dim, PoolingLayer::getOutputDim(input_dim, kernel_size, padding, stride, pool_mode)),
          pool_mode(pool_mode),
          kernel_size(kernel_size), padding(padding), stride(stride) {
        CHECK_MIO(miopenCreatePoolingDescriptor(&desc));
        CHECK_MIO(miopenSet2dPoolingDescriptor(desc, pool_mode, kernel_size, kernel_size, padding, padding, stride, stride));
    }

    ~PoolingLayer() {
        CHECK_MIO(miopenDestroyPoolingDescriptor(desc));
    }

    virtual void init_forward(const Tensor&, Tensor&) override {
        size_t size;
        CHECK_MIO(miopenPoolingGetWorkSpaceSize(output_desc.desc, &size));
        indeces_buf = DevBuffer(size);
    }

    virtual void forward(const Tensor& input, Tensor& output) override {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenPoolingForward(mio::handle(), desc, &alpha, input.desc, input.data, &beta, output.desc, output.data, true, indeces_buf.data, indeces_buf.size));
        // save for backward
        this->input = &input;
        this->output = &output;
    }

    virtual void backward(const Tensor& doutput, Tensor& dinput) override {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenPoolingBackward(mio::handle(), desc, &alpha, getOutputDesc().desc, output->data, doutput.desc, doutput.data, getInputDesc().desc, input->data, &beta, dinput.desc, dinput.data, indeces_buf.data));
    }
};

struct MaxPool : public PoolingLayer {
    MaxPool(const TensorDesc& input_dim, int kernel_size, int padding, int stride)
        : PoolingLayer(input_dim, kernel_size, padding, stride, miopenPoolingMax) {}
};

struct AvgPool : public PoolingLayer {
    AvgPool(const TensorDesc& input_dim, int kernel_size, int padding, int stride)
        : PoolingLayer(input_dim, kernel_size, padding, stride, miopenPoolingAverage) {}
};

struct ReLU : public Layer {
    miopenActivationDescriptor_t desc;

    const Tensor* input_ref;
    const Tensor* output_ref;


    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "ReLU()";
    }

    ReLU(const TensorDesc& input_dim) : Layer(input_dim, input_dim) {
        CHECK_MIO(miopenCreateActivationDescriptor(&desc));
        CHECK_MIO(miopenSetActivationDescriptor(desc, miopenActivationRELU, 0.0, 0.0, 1.0));
    }


    ~ReLU() {
        CHECK_MIO(miopenDestroyActivationDescriptor(desc));
    }

    void forward(const Tensor& input, Tensor& output) {
        float alpha = 1.f;
        float beta = 0.f;
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


struct BatchNorm : public Layer {
    // size of internal tensors (spatial: 1C11, per activation: 1CHW)
    miopenBatchNormMode_t bn_mode;
    TensorDesc bn_dim;

    Tensor scale;
    Tensor dscale;
    Tensor bias;
    Tensor dbias;
    double exp;
    Tensor running_mean;
    Tensor running_var;
    double epsilon;
    Tensor saved_mean; // saved mean for backward
    Tensor saved_ivar; // saved inverse variance for backward

    const Tensor* input_ref; // save reference to input for backward pass

    static TensorDesc get_bn_dim(const TensorDesc& input_dim, miopenBatchNormMode_t bn_mode) {
        TensorDesc bn(0,0,0,0);
        CHECK_MIO(miopenDeriveBNTensorDescriptor(bn.desc, input_dim.desc, bn_mode));
        bn.update_get();
        return bn;
    }

    BatchNorm(const TensorDesc& input_dim, miopenBatchNormMode_t bn_mode=miopenBNSpatial, double eps = 1e-05, double momentum = 0.1)
        : Layer(input_dim, input_dim),
          bn_mode(bn_mode),
          bn_dim(get_bn_dim(input_dim, bn_mode)),
          scale(bn_dim),
          dscale(bn_dim),
          bias(bn_dim),
          dbias(bn_dim),
          exp(momentum),
          running_mean(bn_dim),
          running_var(bn_dim),
          epsilon(eps),
          saved_mean(bn_dim),
          saved_ivar(bn_dim)
    {
    }

    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "BatchNorm()";
    }

    void forward(const Tensor& input, Tensor& output) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenBatchNormalizationForwardTraining(mio::handle(),
                 bn_mode,
                 &alpha,
                 &beta,
                 input.desc,
                 input.data,
                 output.desc,
                 output.data,
                 bn_dim.desc,
                 scale.data,
                 bias.data,
                 exp,
                 running_mean.data,
                 running_var.data,
                 epsilon,
                 saved_mean.data,
                 saved_ivar.data));
        input_ref = &input;
    }

    void backward(const Tensor& doutput, Tensor& dinput) {
        float alpha = 1.f;
        float beta = 0.f;
        CHECK_MIO(miopenBatchNormalizationBackward(mio::handle(),
                     bn_mode,
                     &alpha, 
                     &beta,
                     &alpha,
                     &beta,
                     input_ref->desc,
                     input_ref->data,
                     doutput.desc,
                     doutput.data,
                     dinput.desc,
                     dinput.data,
                     bn_dim.desc,
                     scale.data,
                     dscale.data,
                     dbias.data,
                     epsilon,
                     saved_mean.data,
                     saved_ivar.data));
    }
};

struct Reshape : public Layer {

    Reshape(const TensorDesc& input_dim, int n, int c, int h, int w)
        : Layer(input_dim, TensorDesc(n, c, h, w)) {
        assert(input_dim.n == n);
        assert(input_dim.c * input_dim.h * input_dim.w == c*h*w);
    }

    void init_forward(const Tensor& input, Tensor& output) override {
        output = input.viewAs(getOutputDesc());
    }

    void forward(const Tensor& input, Tensor& output) override {
        output = input.viewAs(getOutputDesc());
    }

    void init_backward(const Tensor& doutput, Tensor& dinput) override {
        dinput = doutput.viewAs(getInputDesc());
    }

    void backward(const Tensor& doutput, Tensor& dinput) override {
        dinput = doutput.viewAs(getInputDesc());
    }
};



#endif // LAYERS_HPP

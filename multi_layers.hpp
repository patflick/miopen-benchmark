#ifndef MULTI_LAYERS_HPP
#define MULTI_LAYERS_HPP

#include <hip/hip_runtime.h>

#include "tensor.hpp"
#include "function.hpp"
#include "utils.hpp"

#include <vector>
#include <memory>

struct Sequential : public Function {
    std::string name;
    TensorDesc input_desc;
    std::vector<std::shared_ptr<Function>> layers;
    std::vector<std::shared_ptr<Tensor>> out_tensors; // the inner buffers

    Sequential(const TensorDesc& input_dim, const std::string& name) : name(name), input_desc(input_dim) {}
    Sequential(const TensorDesc& input_dim) : Sequential(input_dim, "Sequential") {}
    Sequential(const Sequential&) = default;
    Sequential(Sequential&&) = default;

    virtual std::ostream& write_name(std::ostream& os) const {
        return os << name;
    }

    std::string get_name() const {
        return this->name;
    }

    const TensorDesc& last_output_dim() const {
        if (layers.empty()) {
            return input_desc;
        } else {
            return layers.back()->getOutputDesc();
        }
    }

    virtual const TensorDesc& getInputDesc() const override {
        return input_desc;
    }

    virtual const TensorDesc& getOutputDesc() const override {
        return last_output_dim();
    }

    // Calls the LayerType constructor with the input dimension as first argument
    // and then the given arguments LayerType(input_dim, args...);
    template <typename LayerType, typename... Args>
    void emplace(Args... args) {
        if (!layers.empty()) {
            out_tensors.emplace_back(new Tensor(layers.back()->getOutputDesc()));
        }
        layers.emplace_back(new LayerType(last_output_dim(), args...));
    }

    template <typename LayerType>
    void add(const LayerType& l) {
        if (!layers.empty()) {
            out_tensors.emplace_back(new Tensor(layers.back()->getOutputDesc()));
        }
        layers.emplace_back(new LayerType(l));
    }

    template <typename LayerType>
    void add(LayerType&& l) {
        if (!layers.empty()) {
            out_tensors.emplace_back(new Tensor(layers.back()->getOutputDesc()));
        }
        layers.emplace_back(new typename std::remove_reference<LayerType>::type(std::move(l)));
    }

    void addConv(int output_channels, int kernel_size, int padding, int stride) {
        emplace<ConvLayer>(output_channels, kernel_size, padding, stride);
    }

    void addReLU() {
        emplace<ReLU>();
    }

    void addMaxPool(int kernel_size, int padding, int stride) {
        emplace<MaxPool>(kernel_size, padding, stride);
    }

    void addLinear(int outsize) {
        emplace<Linear>(outsize);
    }

    void reshape(int n, int c, int h, int w) {
        emplace<Reshape>(n,c,h,w);
        //out_tensors.emplace_back(new Tensor(n, c, h, w, false)); /* Tensor data gets set in forward() */
    }

    // for each layer, calls f(Layer& l, Tensor& in, Tensor& out);
    template <typename Func>
    void forward_pass(const Tensor& input, Tensor& output, Func f) {
        assert(layers.size() > 0);
        const Tensor* in = &input;
        Tensor* out;

        for (size_t i = 0; i < layers.size(); ++i) {
            if (i < layers.size()-1) {
                out = out_tensors[i].get();
            } else {
                out = &output;
            }
            f(*layers[i], *in, *out);
            in = out;
        }
    }

    // for each layer backwards, calls b(Layer& l, Tensor& dout, Tensor& din)
    template <typename Func>
    void backward_pass(const Tensor& doutput, Tensor& dinput, Func b) {
        assert(layers.size() > 0);
        const Tensor* dout = &doutput;
        Tensor* din;
        for (size_t i = 0; i < layers.size(); ++i) {
            if (i < layers.size()-1) {
                din = out_tensors[layers.size()-i-2].get();
            } else {
                din = &dinput;
            }
            b(*layers[layers.size()-i-1], *dout, *din);
            dout = din;
        }
    }

    // initializes all layers for fwd
    virtual void init_forward(const Tensor& in, Tensor& out) override {
        forward_pass(in, out, [](Function& l, const Tensor& i, Tensor& o){
            l.init_forward(i, o);
        });
    }

    virtual void forward(const Tensor& in, Tensor& out) override {
        forward_pass(in, out, [](Function& l, const Tensor& i, Tensor& o){
            BenchmarkLogger::instance().tic();
            l.forward(i, o);
            BenchmarkLogger::instance().toc(l, false);
        });
    }

    virtual void init_backward(const Tensor& dout, Tensor& din) override {
        backward_pass(dout, din, [](Function& l, const Tensor& o, Tensor& i){
            l.init_backward(o, i);
        });
    }

    virtual void backward(const Tensor& dout, Tensor& din) override {
        backward_pass(dout, din, [](Function& l, const Tensor& o, Tensor& i) {
            BenchmarkLogger::instance().tic();
            l.backward(o, i);
            BenchmarkLogger::instance().toc(l, true);
        });
    }
};


struct Model : public Sequential {

    Tensor input;
    Tensor output;
    bool is_init_fwd;
    bool is_init_bwd;

    Model(const TensorDesc& input_dim, const std::string& name) : Sequential(input_dim, name), input(input_dim), is_init_fwd(false), is_init_bwd(false) {}
    Model(const TensorDesc& input_dim) : Model(input_dim, "Model") {}
    Model(const Model&) = default;
    Model(Model&&) = default;

    using Sequential::init_forward;
    using Sequential::init_backward;
    using Sequential::forward;
    using Sequential::backward;

    void init_forward() {
        if (output.data_size == 0)
            output = Tensor(this->getOutputDesc());
        this->init_forward(input, output);
        is_init_fwd = true;
    }

    void forward() {
        if (!is_init_fwd) {
            init_forward();
        }
        this->forward(input, output);
    }

    void init_backward() {
        if (!is_init_fwd) {
            init_forward();
        }
        this->init_backward(output, input);
        is_init_bwd = true;
    }

    void backward() {
        if (!is_init_bwd) {
            init_backward();
        }
        this->backward(output, input);
    }
};


// implements x += y
/*
void add_inplace(Tensor& x, const Tensor& y) {
    float alpha1 = 1.f, alpha2 = 1.f, beta = 0.f;
    miopenOpTensor(mio::handle(), miopenTensorOpAdd, &alpha1, x.desc, x.data, &alpha2, y.desc, y.data, &beta, x.desc, x.data);
}
*/

__global__ void addinplace_kernel(hipLaunchParm lp, float* x, const float* y, size_t N) {
    size_t offset = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
    size_t stride = hipBlockDim_x * hipGridDim_x;

    for (size_t i = offset; i < N; i+= stride) {
        x[i] = x[i] + y[i];
    }
}

void add_inplace(Tensor& x, const Tensor& y) {
    unsigned int blocks = 512;
    unsigned int threadsPerBlock = 256;
    assert(x.data_size == y.data_size);
    hipLaunchKernel(addinplace_kernel, dim3(blocks), dim3(threadsPerBlock), 0, 0, (float*)x.data, (float*)y.data, x.data_size/4);
}

struct ShortCutAdd : public Function {
    // Implements Residual Shortcutting: y = F(x) + x
    //   where F(x) is any Function with matching input and output dimensions
    // Forward and backward are symmetric in this specific case when addition
    // is used as the combination:
    //   forward(in,out):
    //       out = F.fwd(in) + in (elementwise add)
    //   backward(dout, din):
    //       din = F.bwd(dout) + dout

    TensorDesc input_desc;
    std::shared_ptr<Function> F;
    // optional function for second path
    std::shared_ptr<Function> G;
    // buffers for G outputs
    Tensor gout;
    Tensor gdin;

    ShortCutAdd(const TensorDesc& input_dim) : input_desc(input_dim) {
    }
    ShortCutAdd(const ShortCutAdd&) = default;
    ShortCutAdd(ShortCutAdd&&) = default;

    virtual std::ostream& write_name(std::ostream& os) const {
        return os << "ShortCut";
    }

    template <typename Func>
    void setF(Func f) {
        F = std::shared_ptr<Function>(new typename std::remove_reference<Func>::type(std::forward<Func>(f)));
    }

    template <typename Func>
    void setG(Func g) {
        G = std::shared_ptr<Function>(new typename std::remove_reference<Func>::type(std::forward<Func>(g)));
        gout = Tensor(G->getOutputDesc());
        gdin = Tensor(input_desc);
    }

    virtual const TensorDesc& getInputDesc() const override {
        return F->getOutputDesc();
    }

    virtual const TensorDesc& getOutputDesc() const override {
        assert(F.get() != nullptr);
        return F->getOutputDesc();
    }

    virtual void forward(const Tensor& in, Tensor& out) override {
        assert(F.get() != nullptr);
        BenchmarkLogger::instance().tic();
        F->forward(in, out);
        BenchmarkLogger::instance().toc("ShortcutF", false);
        if (G.get() != nullptr) {
            BenchmarkLogger::instance().tic();
            G->forward(in, gout);
            BenchmarkLogger::instance().toc("ShortcutG", false);
            BenchmarkLogger::instance().tic();
            add_inplace(out, gout);
            BenchmarkLogger::instance().toc("AddInplace", false);
        } else {
            BenchmarkLogger::instance().tic();
            add_inplace(out, in);
            BenchmarkLogger::instance().toc("AddInplace", false);
        }
    }

    virtual void init_forward(const Tensor& in, Tensor& out) {
        F->init_forward(in, out);
        if (G.get() != nullptr) {
            G->init_forward(in, gout);
        }
    }

    virtual void backward(const Tensor& dout, Tensor& din) override {
        BenchmarkLogger::instance().tic();
        F->backward(dout, din);
        BenchmarkLogger::instance().toc("ShortcutF", true);
        if (G.get() != nullptr) {
            BenchmarkLogger::instance().tic();
            G->backward(dout, gdin);
            BenchmarkLogger::instance().toc("ShortcutG", true);
            BenchmarkLogger::instance().tic();
            add_inplace(din, gdin);
            BenchmarkLogger::instance().toc("AddInplace", true);
        } else {
            BenchmarkLogger::instance().tic();
            add_inplace(din, dout);
            BenchmarkLogger::instance().toc("AddInplace", true);
        }
    }

    virtual void init_backward(const Tensor& dout, Tensor& din) override {
        F->init_backward(dout, din);
        if (G.get() != nullptr) {
            G->init_backward(dout, gdin);
        }
    }
};

#endif // MULTI_LAYERS_HPP

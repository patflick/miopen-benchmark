#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <assert.h>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

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

    void resize(size_t new_size) {
        free();
        data = device_alloc(new_size);
        size = new_size;
    }

    ~DevBuffer() {
        free();
    }
};

// static holder of a shared workspace buffer
struct WorkSpace {
    // always returns a device buffer at least the size given
    // if the buffer isn't big enough, its resized 
    static DevBuffer& get() {
        static DevBuffer b;
        return b;
    }
    static DevBuffer& get(size_t size) {
        DevBuffer& b = WorkSpace::get();
        if (b.size < size) {
            DEBUG(" >>> Resizing workspace " << b.size << " -> " << size);
            b.resize(size);
        }
        return b;
    }
};

// 4D Dimensions as NCHW
struct Dim {
    int n;
    int c;
    int h;
    int w;

    Dim() : n(0), c(0), h(0), w(0) {}
    Dim(int n, int c, int h, int w) : n(n), c(c), h(h), w(w) {}
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

    // updates the `Dim` fields by reading the descriptor `desc` with Get4dTensorDescriptor
    void update_get() {
        miopenDataType_t dt;
        int ns, cs, hs, ws;
        CHECK_MIO(miopenGet4dTensorDescriptor(desc, &dt, &n, &c, &h, &w, &ns, &cs, &hs, &ws));
        assert(dt == miopenFloat);
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
          data(o.data),
          data_size(o.data_size),
          owns_data(o.owns_data)
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

    void print_data() {
        std::vector<float> hostTensor = toHost();
        assert(h == 1 && w == 1); // current limitation
        assert(hostTensor.size() == (size_t)n*c);
        std::cout << "Tensor of size " << *this << ":" << std::endl << "[";
        for (int i = 0; i < n; ++i) {
            if (i > 0)
                std::cout << " ";
            std::cout << "[";
            for (int j = 0; j < c; ++j) {
                std::cout << hostTensor[i*n + j];
                if (j+1 < c)
                    std::cout << ", ";
            }
            if (i+1 < n)
                std::cout << "]," << std::endl;
            else
                std::cout << "]]" << std::endl;
        }
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
          data_size(n*(size_t)c*h*w*sizeof(float)),
          owns_data(true) {
        alloc();
    }

    Tensor(const Dim& dims)
        : TensorDesc(dims),
          data_size(n*(size_t)c*h*w*sizeof(float)),
          owns_data(true) {
        alloc();
    }

    Tensor(int n, int c, int h, int w)
        : TensorDesc(n, c, h, w),
          data_size(n*(size_t)c*h*w*sizeof(float)),
          owns_data(true) {
        alloc();
    }

    Tensor(int n, int c, int h, int w, bool do_alloc)
        : TensorDesc(n, c, h, w),
          data_size(n*(size_t)c*h*w*sizeof(float)),
          owns_data(do_alloc) {
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

#endif // TENSOR_HPP

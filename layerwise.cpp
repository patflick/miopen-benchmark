

#include "miopen.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"


void benchmark_convlayers() {
    // batch_size, w, h, channels_in, channels_out, kernel_size, padding, stride
    // Layerwise benchmark L1-L5: https://github.com/soumith/convnet-benchmarks
    std::vector<ConvLayerDesc> runs = {{128, 13, 13, 384, 384, 3, 0, 1},
                                   {128, 16, 16, 128, 128, 7, 0, 1},
                                   {128, 32, 32, 128, 128, 9, 0, 1},
                                   {128, 64, 64, 64, 128, 9, 0, 1},
                                   {128, 128, 128, 3, 96, 11, 0, 1}};


    /*
    std::vector<ConvLayerDesc> runs = {{128, 64, 64, 64, 128, 3, 1, 1}};
                                       {128, 64, 64, 64, 128, 3, 0, 1},
                                       {128, 28, 28, 64, 64, 5, 1, 2}};
                                       */


    int layer = 5;
    int reps = 50;
    BenchmarkLogger::new_session("conv_layers");
    for (ConvLayerDesc& l : runs) {
        std::stringstream ss;
        ss << "Layer L" << layer;
        TensorDesc input_dim(l.batch_size, l.channels_in, l.height, l.width);
        Model m(input_dim, ss.str());
        m.emplace<ConvLayer>(l.channels_out, l.kernel_size, l.padding, l.stride);

        BenchmarkLogger::benchmark(m, reps);

        --layer;
    }
}

int main(int argc, char *argv[])
{
    device_init();
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    benchmark_convlayers();
}

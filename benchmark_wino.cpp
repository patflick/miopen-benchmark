#include "miopen.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"

#include <cstdlib>

std::map<std::string, ConvLayerDesc> get_layers(){
    std::map<std::string, ConvLayerDesc> m;
    // batch_size, w, h, channels_in, channels_out, kernel_size, padding, stride
    m.emplace("L2", ConvLayerDesc({128, 64, 64, 64, 128, 9, 0, 1}));
    m.emplace("W1", ConvLayerDesc({128, 64, 64, 64, 128, 3, 1, 1}));
    return m;
}

std::map<std::string, ConvLayerDesc>& layers() {
    static std::map<std::string, ConvLayerDesc> m = get_layers();
    return m;
}

int main(int argc, char *argv[])
{
    device_init();
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    std::string layer_name = "W1";
    if (argc >= 2) {
        layer_name = argv[1];
        if (layers().count(layer_name) == 0) {
            FATAL("Unknown layer name `" << layer_name << "`.");
        }
    }
    int reps = 1000;
    if (argc >= 3) {
        reps = atoi(argv[2]);
        if (reps <= 0) {
            FATAL("Bad iteration count: `" << std::string(argv[2]) << "`. Has to be int and > 0");
        }
    }


    // create model of single layer
    ConvLayerDesc l = get_layers()[layer_name];
    TensorDesc input_dim(l.batch_size, l.channels_in, l.height, l.width);
    Model m(input_dim);
    m.emplace<ConvLayer>(l.channels_out, l.kernel_size, l.padding, l.stride);

    // benchmark model forward
    BenchmarkLogger::new_session(layer_name);
    BenchmarkLogger::fwd_layer_benchmark(m, reps);
    return 0;
}

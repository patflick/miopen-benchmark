
#include "miopen.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"

int main(int argc, char *argv[])
{
    device_init();
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    // batch_size, w, h, channels_in, channels_out, kernel_size, padding, stride
    ConvLayerDesc l({128, 64, 64, 64, 128, 3, 1, 1});
    //ConvLayerDesc l({128, 64, 64, 64, 128, 9, 0, 1});
    TensorDesc input_dim(l.batch_size, l.channels_in, l.height, l.width);
    Model m(input_dim);
    m.emplace<ConvLayer>(l.channels_out, l.kernel_size, l.padding, l.stride);

    // benchmark fwd
    BenchmarkLogger::new_session("wino_conv");
    BenchmarkLogger::fwd_layer_benchmark(m, 100000);

    return 0;
}

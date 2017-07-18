#include "miopen.hpp"
#include "tensor.hpp"
#include "utils.hpp"
#include "layers.hpp"
#include "multi_layers.hpp"


/* TODO ResNet
 * - [x] ShortCut module
 * - [x] tensor elementwise addition
 * - [ ] BatchNorm
 */

Sequential makeBlock(const TensorDesc& input_dim, int planes, int stride=1, bool downsample = false) {
    // BasicBlock
    DEBUG("making block with dim " << input_dim );
    Sequential preblock(input_dim);
    preblock.emplace<ConvLayer>(planes, 3, 1, stride);
    //block.emplace<BatchNorm>(outplanes) TODO BatchNorm
    preblock.emplace<ReLU>();
    preblock.emplace<ConvLayer>(planes, 3, 1, 1);
    //block.emplace<BatchNorm>(outplanes) TODO BatchNorm


    Sequential down_block(input_dim);
    down_block.emplace<ConvLayer>(planes, 1, 0, stride);
    // TODO: batchnorm for down_block

    Sequential block(input_dim);
    ShortCutAdd s(input_dim);
    s.setF(preblock);
    if (downsample) {
        s.setG(down_block);
    }
    block.add(s);
    block.emplace<ReLU>();
    return block;
}

Sequential makeLayer(const TensorDesc& input_dim, int planes, int blocks, int stride=1) {
    Sequential layer(input_dim);

    // add one downsample block inplanes -> planes, stride
    bool downsample = stride != 1;
    layer.add(makeBlock(layer.getOutputDesc(), planes, stride, downsample));

    for (int i = 0; i < blocks; ++i) {
        layer.add(makeBlock(layer.getOutputDesc(), planes));
    }
    return layer;
}

void resnet() {

    TensorDesc input_dim(128, 3, 224, 224);

    Model m(input_dim);

    Sequential pre(input_dim);
    pre.emplace<ConvLayer>(64, 7, 3, 2);
    // TODO batch norm
    pre.emplace<ReLU>();
    pre.emplace<MaxPool>(3, 0, 2);
    DEBUG("ResNet Pre output dims: " << pre.getOutputDesc());

    m.add(pre);

    // ResNet 18
    std::vector<int> layers = {2, 2, 2, 2};

    m.add(makeLayer(m.getOutputDesc(), 64, layers[0]));
    m.add(makeLayer(m.getOutputDesc(), 128, layers[1], 2));
    m.add(makeLayer(m.getOutputDesc(), 256, layers[2], 2));
    m.add(makeLayer(m.getOutputDesc(), 512, layers[3], 2));
    m.emplace<AvgPool>(7, 0, 1);

    BenchmarkLogger::benchmark(m);
}

int main(int argc, char *argv[])
{
    device_init();
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    resnet();
}

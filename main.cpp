

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

/* TODO:
 * - [ ] create AlexNet class
 * - [ ] uniform random tensors (via host->device copy), and CPU initialized tensors
 * - [x] Make `Model` take input and output tensors in forward(), backward()
 * - [ ] Collect total and average times per layer
 * - [ ] implement and benchmark ResNet
 */

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

        Model m(conv.getInputDesc());
        m.add(conv);
        //Tensor input(conv.getInputDesc());
        //Tensor output(conv.getOutputDesc());
        //
        BenchmarkLogger bm("benchmark.log");
        bm.benchmark(m);

#if 0
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
#endif
        --layer;
    }

}


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

    // TODO
    //benchmarkModel(m);
}

void alexNet() {
    TensorDesc input_dim(128, 3, 224, 224);

    Sequential features(input_dim);
    /* features */
    features.addConv(64, 11, 2, 4);
    features.addReLU();
    features.addMaxPool(3, 0, 2);
    features.addConv(192, 5, 2, 1);
    features.addReLU();
    features.addMaxPool(3, 0, 2);
    features.addConv(384, 3, 1, 1);
    features.addReLU();
    features.addConv(256, 3, 1, 1);
    features.addReLU();
    features.addConv(256, 3, 1, 1);
    features.addReLU();
    features.addMaxPool(3, 0, 2);

    DEBUG("Dims after Features: " << features.getOutputDesc());

    /* classifier */
    Sequential classifier(features.getOutputDesc());
    // TODO Dropout
    classifier.reshape(128, 256 * 6 * 6, 1, 1);
    classifier.addLinear(4096);
    classifier.addReLU();
    // TODO: Dropout
    classifier.addLinear(4096);
    classifier.addReLU();
    classifier.addLinear(1000);

    Model m(input_dim);
    m.add(features);
    m.add(classifier);

    BenchmarkLogger::instance().benchmark(m);
    //benchmarkModel(m);
}

void check_add() {
    Tensor x(2, 2, 1, 1);
    x.fromHost({3, 4, 2, 1});
    x.print_data();

    Tensor y(2, 2, 1, 1);
    y.fromHost({-3, .15, 2, 5});
    y.print_data();

    add_inplace(x, y);
    x.print_data();
}

int main(int argc, char *argv[])
{
    device_init();

    // enable profiling
    CHECK_MIO(miopenEnableProfiling(mio::handle(), true));

    alexNet();
    //benchmark_convlayers();
    //resnet();

    miopenDestroy(mio::handle());
    return 0;
}

#ifndef MY_MIOPEN_HPP
#define MY_MIOPEN_HPP

#include <miopen/miopen.h>
#include <hipblas.h>
//#include <gperftools/profiler.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

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
#define INFO(msg) std::cout << "[INFO]  " << msg << std::endl;

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
#endif // MY_MIOPEN_HPP

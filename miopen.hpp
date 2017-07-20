#ifndef MY_MIOPEN_HPP
#define MY_MIOPEN_HPP

#include <miopen/miopen.h>
#include <hipblas.h>
//#include <gperftools/profiler.h>

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <regex>
#include <dirent.h>

//#define WITH_CL

#define DEBUG(msg) std::cerr << "[DEBUG] " << msg << std::endl;
#define WARNING(msg) std::cerr << "[!WARNING!] " << msg << std::endl;
#define FATAL(msg) {std::cerr << "[!FATAL!] " << msg << std::endl; exit(EXIT_FAILURE);} while(0)
#define INFO(msg) std::cout << "[INFO]  " << msg << std::endl;

#ifdef WITH_CL // not supported anymore
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

std::vector<std::string> split(const std::string& str, char sep) {
    std::vector<std::string> strings;
    std::istringstream f(str);
    std::string s;
    while (std::getline(f, s, sep)) {
        strings.push_back(s);
    }
    return strings;
}

void print_file(const std::string& fname) {
    std::ifstream f(fname);
    std::string s;
    std::cout << "File: " << fname << ", status: " << f.good() << std::endl;
    while (std::getline(f, s)) {
        std::cout << ":: " << s << std::endl;
    }

}

std::vector<std::string> ls_dir(const std::string& dname) {
    std::vector<std::string> files;
    struct dirent* entry;
    DIR *dir = opendir(dname.c_str());
    if (dir == NULL) {
        return files;
    }

    while ((entry = readdir(dir)) != NULL) {
        std::string fname(entry->d_name);
        if (fname != "." && fname != "..")
            files.push_back(fname);
    }
    return files;
}

std::vector<std::string> ls_dir(const std::string& dname, const std::regex& match) {
    std::vector<std::string> files;
    struct dirent* entry;
    DIR *dir = opendir(dname.c_str());
    if (dir == NULL) {
        return files;
    }

    while ((entry = readdir(dir)) != NULL) {
        std::string fname(entry->d_name);
        if (fname != "." && fname != "..") {
            if (std::regex_match(fname, match)) {
                files.push_back(fname);
            }
        }
    }
    return files;
}

int read_current_mhz(const std::string& fname) {
    std::ifstream f(fname);
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
    return -1;
}

struct Device {
    int hip_id;
    hipDeviceProp_t hip_props;
    std::string drm_path;
    std::string hwmon_path;

    /// Find the sysfs paths given that the device is initalized by hipGetDeviceProperties
    /// The paths are found using hiDeviceProp_t::pciBusID
    void init_sys_paths() {
        bool found = false;
        for (std::string cardname : ls_dir("/sys/class/drm", std::regex("card\\d+"))) {
            std::string carddir = "/sys/class/drm/" + cardname;
            std::string fname = carddir + "/device/uevent";
            std::ifstream f(fname);
            if (f.good() && f.is_open()) {
                std::string line;
                while (std::getline(f, line)) {
                    if (split(line, '=')[0] == "PCI_SLOT_NAME") {
                        std::string pciids = split(line, '=')[1];
                        std::vector<std::string> ids = split(pciids, ':');
                        //std::string busid = "0x" + ids[1];
                        int pci_busid = std::stoul("0x" + ids[1], nullptr, 16);
                        if (pci_busid == hip_props.pciBusID) {
                            drm_path = carddir;
                            // find hwmon path
                            std::vector<std::string> hwpaths = ls_dir(drm_path + "/device/hwmon", std::regex("hwmon\\d+"));
                            if (hwpaths.size() != 1) {
                                WARNING("No or multiple hwmon paths for " << drm_path);
                            }
                            if (hwpaths.size() > 0) {
                                hwmon_path = drm_path + "/device/hwmon/" + hwpaths[0];
                            }
                            found = true;
                        }
                        break;
                    }
                }
            }
            if (found)
                break;
        }
        if (!found) {
            WARNING("Can't find sysfs path for device " << hip_id);
        }
    }

    void print_info() {
        // print out device info
        INFO("Device " << hip_id << ": " << hip_props.name);
        INFO("\tArch:\t" << hip_props.gcnArch)
        INFO("\tGMem:\t" << hip_props.totalGlobalMem/1024/1024 << " MiB");
        INFO("\twarps:\t" << hip_props.warpSize);
        INFO("\tCUs:\t" << hip_props.multiProcessorCount);
        INFO("\tMaxClk:\t" << hip_props.clockRate);
        INFO("\tMemClk:\t" << hip_props.memoryClockRate);
        INFO("\tdrm:\t" << drm_path);
        INFO("\thwmon:\t" << hwmon_path);
        //INFO("\t\tpciDomainID:\t" << hip_props.pciDomainID);
        //INFO("\t\tpciBusID:\t" << hip_props.pciBusID);
        //INFO("\t\tpciDeviceID:\t" << hip_props.pciDeviceID);
#ifdef __HIP_PLATFORM_HCC__
#endif
    }

    float getTemp() {
        std::ifstream f(hwmon_path + "/temp1_input");
        //std::ifstream f("/sys/class/hwmon/hwmon0/temp1_input");
        int temp;
        f >> temp;
        return temp / 1000.f; // temp is in milli celsius
    }

    int getFanspeed() {
        //std::ifstream f("/sys/class/hwmon/hwmon0/pwm1");
        std::ifstream f(hwmon_path + "/pwm1");
        int fan;
        f >> fan;
        return fan;
    }

    int getClock() {
        return read_current_mhz(drm_path + "/device/pp_dpm_sclk");
    }

    int getMemClock() {
        return read_current_mhz(drm_path + "/device/pp_dpm_mclk");
    }
};

struct Devices {
    static std::vector<Device>& get_devices(bool from_init = false) {
        static bool is_init = false;
        static std::vector<Device> d;
        if (!is_init) {
            is_init = true;
            if (!from_init)
                init_devices();
        }
        return d;
    }

    static Device& get_default_device() {
        if (get_devices().size() == 0) {
            FATAL("No HIP Devices available.");
        }
        return get_devices()[0];
    }

    static void init_devices() {
        int devcount;
        CHECK_HIP(hipGetDeviceCount(&devcount));
        INFO("Number of HIP devices found: " << devcount);

        if (devcount == 0) {
            FATAL("No HIP devices found.");
        }

        std::vector<Device>& devs = get_devices(true);
        devs.resize(devcount);

        // init and get devices
        for (int d = 0; d < devcount; ++d) {
            devs[d].hip_id = d;
            CHECK_HIP(hipGetDeviceProperties(&devs[d].hip_props, d/*deviceID*/));
            devs[d].init_sys_paths();
            devs[d].print_info();
        }

    }

};


void device_init() {
    Devices::init_devices();
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
    // This is called once, the first time the MIOpen handle is retrieved
    static miopenHandle_t init_mio() {
        miopenHandle_t h;
        CHECK_HIP(hipSetDevice(Devices::get_default_device().hip_id));
        hipStream_t q;
        CHECK_HIP(hipStreamCreate(&q));
        CHECK_MIO(miopenCreateWithStream(&h, q));
        return h;
    }
public:
    static miopenHandle_t handle() {
        static miopenHandle_t h = init_mio();
        return h;
    }
};


float getTemp() {
    return Devices::get_default_device().getTemp();
}

int getFanspeed() {
    return Devices::get_default_device().getFanspeed();
}

int getClock() {
    return Devices::get_default_device().getClock();
}

int getMemClock() {
    return Devices::get_default_device().getMemClock();
}

#endif // MY_MIOPEN_HPP

#ifndef UTILS_HPP
#define UTILS_HPP

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <chrono>
#include <vector>

#include "miopen.hpp"
#include "function.hpp"


// milliseconds precision timer
struct Timer {
    using duration = std::chrono::steady_clock::duration;
    using time_point = std::chrono::steady_clock::time_point;

    time_point tic_tp;
    time_point toc_tp;

    Timer() : tic_tp(std::chrono::steady_clock::now()) {}

    void tic() {
        tic_tp = std::chrono::steady_clock::now();
    }

    float toc() {
        time_point toc_tp = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(toc_tp - tic_tp).count() / 1000.f;
    }
};

struct layertimer {
    std::string name;
    using duration = std::chrono::steady_clock::duration;
    using time_point = std::chrono::steady_clock::time_point;
    duration cum_time;
    std::vector<duration> lap_times;
    time_point tic_tp;
    layertimer(const std::string& name) : name(name), cum_time(0) {
    }

    void tic() {
        tic_tp = std::chrono::steady_clock::now();
    }

    static float to_ms(duration d) {
        return std::chrono::duration_cast<std::chrono::microseconds>(d).count() / 1000.f;
    }

    float toc() {
        time_point toc_tp = std::chrono::steady_clock::now();
        lap_times.emplace_back(toc_tp - tic_tp);
        cum_time += lap_times.back();
        return to_ms(toc_tp - tic_tp);
    }

    duration total_time() const {
        return cum_time;
    }

    float total_time_ms() const {
        return to_ms(cum_time);
    }

    float avg_time_ms() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(cum_time).count() / 1000.f / lap_times.size();
    }

    std::vector<float> times_ms() const {
        std::vector<float> result(lap_times.size());
        for (size_t i = 0; i < lap_times.size(); ++i) {
            result[i] = std::chrono::duration_cast<std::chrono::microseconds>(lap_times[i]).count() / 1000.f;
        }
        return result;
    }
};

struct BenchmarkLogger : public Timer {

    std::ofstream of;

    std::chrono::steady_clock::time_point start_time;

    void init() {
        of << "Timestamp\tModule\tDir\tTime\tTemp\tFan\tClock\tMemClock" << std::endl;
    }

    BenchmarkLogger() : of(), start_time() {}

    BenchmarkLogger(const std::string& filename)
        : of(filename), start_time(std::chrono::steady_clock::now()) {
        init();
    }

    BenchmarkLogger(BenchmarkLogger&&) = default;
    BenchmarkLogger& operator=(BenchmarkLogger&&) = default;

    void log_step(const std::string& module_name, bool bwd, float duration) {
        auto toc = std::chrono::steady_clock::now();
        std::string dir = bwd ? "bwd" : "fwd";
        auto timestamp = std::chrono::duration_cast<std::chrono::microseconds>(toc - start_time).count();
        float temp = getTemp();
        int fan = getFanspeed();
        int clk = getClock();
        int mclk = getMemClock();
        of << timestamp << "\t" << module_name << "\t" << dir << "\t" << duration << "\t" << temp << "\t" << fan << "\t" << clk << "\t" << mclk << std::endl;
        INFO(module_name << ":\t" << duration << " ms");
    }

    static BenchmarkLogger& instance() {
        static BenchmarkLogger bm = BenchmarkLogger();
        return bm;
    }

    static void new_session(const std::string& name = "") {
        static int count = 0;
        ++count;
        std::stringstream ss;
        if (name == "") {
            ss << "benchmark_" << count << ".tsv";
        } else {
            ss << name << ".tsv";
        }
        if (instance().of.is_open())
            instance().of.close();
        instance().of.open(ss.str());
        instance().init();
        instance().start_time = std::chrono::steady_clock::now();
    }

    static void log(const std::string& module_name, bool bwd, float duration) {
        instance().log_step(module_name, bwd, duration);
    }

    using Timer::toc;
    void toc(Function& f, bool bwd) {
#if LAYER_TIMING == 1
        CHECK_HIP(hipDeviceSynchronize());
        float dur = this->toc();
        std::stringstream ss;
        ss << f;
        log_step(ss.str(), bwd, dur);
#endif
    }

    void toc(const std::string& s, bool bwd) {
        CHECK_HIP(hipDeviceSynchronize());
        float dur = this->toc();
        log_step(s, bwd, dur);
    }

    template <typename M>
    static void benchmark(M& m, int reps = 10, bool runbwd = true) {
        instance().do_benchmark(m, reps, runbwd);
    }

    template <typename M>
    void do_fwd_layer_benchmark(M& m, int reps = 10) {
        INFO("Init fwd");
        m.init_forward();
        float layer_time;
        for (int i = 0; i < reps; ++i) {
            m.forward();
            CHECK_MIO(miopenGetKernelTime(mio::handle(), &layer_time));
            log_step("KernelTime", false, layer_time);
        }
    }

    template <typename M>
    static void fwd_layer_benchmark(M& m, int reps = 10) {
        instance().do_fwd_layer_benchmark(m, reps);
    }

    template <typename M>
    void do_benchmark(M& m, int reps = 10, bool runbwd = true) {
        INFO("Init fwd");
        m.init_forward();
        if (runbwd) {
            INFO("Init bwd");
            m.init_backward();
        }

        INFO("Begin warmup runs");
        Timer timer;
        for (int i = 0; i < 1; ++i) {
            {
                INFO("               ======= BEGIN FWD =======");
                timer.tic();
                m.forward();
                CHECK_HIP(hipDeviceSynchronize());
                log_step(m.get_name(), false, timer.toc());
            }
            if (runbwd) {
                INFO("               ======= BEGIN BWD =======");
                timer.tic();
                m.backward();
                CHECK_HIP(hipDeviceSynchronize());
                log_step(m.get_name(), true, timer.toc());
            }
        }

        INFO("Begin Timings");

        layertimer fwdtime("fwd");
        layertimer bwdtime("bwd");
        timer.tic();
        for (int i = 0; i < reps; ++i) {
            {
                INFO("               ======= BEGIN FWD =======");
                fwdtime.tic();
                m.forward();
                CHECK_HIP(hipDeviceSynchronize());
                log_step(m.get_name(), false, fwdtime.toc());
            }
            if (runbwd) {
                INFO("               ======= BEGIN BWD =======");
                bwdtime.tic();
                m.backward();
                CHECK_HIP(hipDeviceSynchronize());
                log_step(m.get_name(), true, bwdtime.toc());
            }
        }
        double time_per = timer.toc()/reps;
        INFO("Avg time per fwd " << fwdtime.avg_time_ms() << " ms");
        INFO("Avg time per bwd " << bwdtime.avg_time_ms() << " ms");
        INFO("Avg time per fwd+bwd: " << time_per << " ms");
    }

};

#endif // UTILS_HPP


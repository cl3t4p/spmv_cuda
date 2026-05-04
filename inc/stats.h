#ifndef SPMV_CUDA_STATS_H
#define SPMV_CUDA_STATS_H

#include <cmath>
#include <cstddef>
#include <vector>

template <typename T> double arithmetic_mean(const std::vector<T> &v) {
    if (v.empty())
        return 0.0;
    double mu = 0.0;
    for (size_t i = 0; i < v.size(); ++i)
        mu += static_cast<double>(v[i]);
    return mu / static_cast<double>(v.size());
}

template <typename T> double geometric_mean(const std::vector<T> &v) {
    if (v.empty())
        return 0.0;
    double log_sum = 0.0;
    size_t n = 0;
    for (size_t i = 0; i < v.size(); ++i) {
        const double x = static_cast<double>(v[i]);
        if (x > 0.0) {
            log_sum += std::log(x);
            ++n;
        }
    }
    return (n > 0) ? std::exp(log_sum / static_cast<double>(n)) : 0.0;
}

template <typename T> double variance(const std::vector<T> &v, double mu) {
    if (v.empty())
        return 0.0;
    double s = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        const double d = static_cast<double>(v[i]) - mu;
        s += d * d;
    }
    return s / static_cast<double>(v.size());
}

template <typename T> double stddev(const std::vector<T> &v, double mu) { return std::sqrt(variance(v, mu)); }

#endif // SPMV_CUDA_STATS_H

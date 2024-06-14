#pragma once

#include <cstddef>
#include <vector>

namespace ATL24_qtrees
{

namespace features
{

struct window_features
{
    double mean;
    double median;
    double variance;
    std::vector<double> quantiles;
};

struct window
{
    size_t begin_photon_index;
    size_t end_photon_index;
    window_features f;
};

struct photon_features
{
    double elevation;
    double density;
    size_t window_index;
};

} // namespace features

} // namespace ATL24_qtrees

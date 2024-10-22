#pragma once

#include "ATL24_qtrees/dataframe.h"

const std::string pi_name ("index_ph");
const std::string x_name ("x_atc");
const std::string z_name ("geoid_corr_h");

namespace ATL24_qtrees
{

namespace utils
{

struct sample
{
    size_t dataset_id;
    size_t h5_index;
    double x;
    double z;
    size_t cls;
    size_t prediction;
    double surface_elevation;
    double bathy_elevation;
};

namespace constants
{
    constexpr double max_photon_elevation = 20.0; // meters
    constexpr double min_photon_elevation = -80.0; // meters
    constexpr float missing_data = std::numeric_limits<float>::max ();
    constexpr unsigned unclassified_class = 1;
    constexpr unsigned bathy_class = 40;
    constexpr unsigned sea_surface_class = 41;
    constexpr double surface_sigma = 100.0; // meters
    constexpr double max_surface_elevation = 20.0; // meters
    constexpr double min_surface_elevation = -20.0; // meters
    constexpr double max_surface_estimate_delta = 10.0; // meters
    constexpr double bathy_sigma = 60.0; // meters
    constexpr double min_bathy_depth = 1.5; // meters
    constexpr double max_bathy_estimate_delta = 10.0; // meters
};

struct feature_params
{
    double window_size = 40.0; // meters
    size_t total_quantiles = 32;
    size_t adjacent_windows = 2;
};

std::ostream &operator<< (std::ostream &os, const feature_params &fp)
{
    os << std::fixed;
    os << std::setprecision(3);
    os << "window_size: " << fp.window_size << std::endl;
    os << "total_quantiles: " << fp.total_quantiles << std::endl;
    os << "adjacent_windows: " << fp.adjacent_windows << std::endl;
    return os;
}

uint32_t remap_label (const uint32_t label)
{
    switch (label)
    {
        default: return 0;
        case 40: return 1;
        case 41: return 2;
    }
}

uint32_t unremap_label (const uint32_t label)
{
    switch (label)
    {
        default: return 0;
        case 1: return 40;
        case 2: return 41;
    }
}

struct window
{
    std::vector<double> quantiles;
};

template<typename T>
std::vector<size_t> get_window_indexes (const T &samples, const double &window_size)
{
    using namespace std;

    const double min_x = min_element (samples.cbegin (), samples.cend (),
        [] (const auto &a, const auto &b) { return a.x < b.x; })->x;

    // Get the index for each photon
    std::vector<size_t> indexes (samples.size ());

#pragma omp parallel for
    for (size_t i = 0; i < samples.size (); ++i)
    {
        // What is the photon's window index?
        const size_t index = (samples[i].x - min_x) / window_size;
        indexes[i] = index;
    }

    return indexes;
}

template<typename T,typename U>
std::vector<double> get_quantiles (T x, const U &fp)
{
    using namespace std;

    vector<double> q (fp.total_quantiles);
    if (x.size () < fp.total_quantiles)
        return q;

    // Order them
    sort (x.begin (), x.end ());

    // Count number in each quantile
    vector<size_t> total (fp.total_quantiles);

    const double photons_per_quantile = static_cast<double> (x.size ()) / fp.total_quantiles;

    for (size_t i = 0; i < x.size (); ++i)
    {
        const size_t index = i / photons_per_quantile;

        assert (index < q.size ());
        assert (index < total.size ());

        q[index] += x[i];
        ++total[index];
    }

    // Get the quantile's average value
    for (size_t i = 0; i < q.size (); ++i)
        q[i] /= total[i];

    return q;
}

template<typename T,typename U>
window create_window (const T &elevations, const U &fp)
{
    using namespace std;

    // Throw out elevations that are out of range
    T e;
    copy_if (elevations.begin (), elevations.end (), back_inserter (e),
            [](double x) {
                return x > constants::min_photon_elevation
                         && x < constants::max_photon_elevation;
            });

    return window {get_quantiles (e, fp)};
}

template<typename T,typename U,typename V>
std::vector<window> get_windows (const T &samples,
    const U &fp,
    const V &window_indexes)
{
    using namespace std;

    // Check logic
    assert (samples.size () == window_indexes.size ());

    // Get the min/max x for all samples
    const size_t max_index = *max_element (window_indexes.begin (), window_indexes.end ());
    const size_t n = max_index + 1;

    // Get the elevations in each window
    std::vector<std::vector<double>> elevations (n);

    for (size_t i = 0; i < samples.size (); ++i)
    {
        const size_t index = (window_indexes[i]);
        assert (index < elevations.size ());
        elevations[index].push_back (samples[i].z);
    }

    std::vector<window> w (n);

#pragma omp parallel for
    for (size_t i = 0; i < w.size (); ++i)
        w[i] = create_window (elevations[i], fp);

    return w;
}

template<typename T>
class features
{
    public:
    features (const T &init_samples, const feature_params &init_fp)
        : samples (init_samples)
        , fp (init_fp)
        , window_indexes (get_window_indexes (init_samples, init_fp.window_size))
        , windows (get_windows (init_samples, init_fp, window_indexes))
    {
        assert (window_indexes.size () == samples.size ());
    }
    size_t features_per_sample () const
    {
        // total features =
        //        photon elevation
        //      + quantiles in photon's window
        //      + quantiles in adjacent windows
        return    1
                + fp.total_quantiles
                + (2 * fp.adjacent_windows) * fp.total_quantiles;
    }
    std::vector<float> get_features (const size_t n) const
    {
        using namespace std;
        using namespace constants;

        // Check invariants
        assert (n < window_indexes.size ());

        // Get the features for this photon
        vector<float> f; f.reserve (features_per_sample ());

        // Elevation
        f.push_back (samples[n].z);

        // Quantiles for the photon's window
        const size_t i = window_indexes[n];
        f.insert (f.end (), windows[i].quantiles.begin (), windows[i].quantiles.end ());

        // Quantiles for adjacent windows
        for (size_t j = 0; j < fp.adjacent_windows; ++j)
        {
            // Push window on the right
            const size_t right_index = i + (j + 1);
            if (right_index < windows.size ())
                f.insert (f.end (), windows[right_index].quantiles.begin (), windows[right_index].quantiles.end ());
            else
                f.insert (f.end (), windows[i].quantiles.size (), missing_data);

            // Push window on the left
            const size_t left_index = i - (j + 1);
            if (left_index < windows.size ())
                f.insert (f.end (), windows[left_index].quantiles.begin (), windows[left_index].quantiles.end ());
            else
                f.insert (f.end (), windows[i].quantiles.size (), missing_data);
        }

        // Check invariants
        assert (f.size () == features_per_sample ());

        return f;
    }
    private:
    const std::vector<sample> &samples;
    feature_params fp;
    std::vector<size_t> window_indexes;
    std::vector<window> windows;
};

template<typename T>
std::vector<sample> convert_dataframe (const T &df)
{
    using namespace std;

    // Check invariants
    assert (df.is_valid ());
    assert (!df.headers.empty ());
    assert (!df.columns.empty ());

    // Get number of photons in this file
    const size_t nrows = df.columns[0].size ();

    // Get the columns we are interested in
    auto pi_it = find (df.headers.begin(), df.headers.end(), pi_name);
    auto x_it = find (df.headers.begin(), df.headers.end(), x_name);
    auto z_it = find (df.headers.begin(), df.headers.end(), z_name);
    auto cls_it = find (df.headers.begin(), df.headers.end(), "manual_label");
    auto prediction_it = find (df.headers.begin(), df.headers.end(), "prediction");
    auto surface_elevation_it = find (df.headers.begin(), df.headers.end(), "sea_surface_h");
    auto bathy_elevation_it = find (df.headers.begin(), df.headers.end(), "bathy_h");

    assert (pi_it != df.headers.end ());
    assert (x_it != df.headers.end ());
    assert (z_it != df.headers.end ());

    if (pi_it == df.headers.end ())
        throw runtime_error ("Can't find 'index_ph' in dataframe");
    if (x_it == df.headers.end ())
        throw runtime_error ("Can't find 'x_atc' in dataframe");
    if (z_it == df.headers.end ())
        throw runtime_error ("Can't find 'geoid_corr_h' in dataframe");

    size_t index_ph = pi_it - df.headers.begin();
    size_t x_index = x_it - df.headers.begin();
    size_t z_index = z_it - df.headers.begin();
    const bool has_manual_label = cls_it != df.headers.end ();
    size_t cls_index = has_manual_label ?
        cls_it - df.headers.begin() :
        df.headers.size ();
    const bool has_predictions = prediction_it != df.headers.end ();
    size_t prediction_index = has_predictions ?
        prediction_it - df.headers.begin() :
        df.headers.size ();
    const bool has_surface_elevations = surface_elevation_it != df.headers.end ();
    size_t surface_elevation_index = has_surface_elevations ?
        surface_elevation_it - df.headers.begin() :
        df.headers.size ();
    const bool has_bathy_elevations = bathy_elevation_it != df.headers.end ();
    size_t bathy_elevation_index = has_bathy_elevations ?
        bathy_elevation_it - df.headers.begin() :
        df.headers.size ();

    // Check logic
    assert (index_ph < df.headers.size ());
    assert (x_index < df.headers.size ());
    assert (z_index < df.headers.size ());
    if (has_manual_label)
        assert (cls_index < df.headers.size ());
    if (has_predictions)
        assert (prediction_index < df.headers.size ());
    if (has_surface_elevations)
        assert (surface_elevation_index < df.headers.size ());
    if (has_bathy_elevations)
        assert (bathy_elevation_index < df.headers.size ());

    assert (index_ph < df.columns.size ());
    assert (x_index < df.columns.size ());
    assert (z_index < df.columns.size ());
    if (has_manual_label)
        assert (cls_index < df.columns.size ());
    if (has_predictions)
        assert (prediction_index < df.columns.size ());
    if (has_surface_elevations)
        assert (surface_elevation_index < df.columns.size ());
    if (has_bathy_elevations)
        assert (bathy_elevation_index < df.columns.size ());

    // Stuff values into the vector
    std::vector<sample> dataset (nrows);

    for (size_t j = 0; j < nrows; ++j)
    {
        // Check logic
        assert (j < df.columns[index_ph].size ());
        assert (j < df.columns[x_index].size ());
        assert (j < df.columns[z_index].size ());
        if (has_manual_label)
            assert (j < df.columns[cls_index].size ());
        if (has_predictions)
            assert (j < df.columns[prediction_index].size ());
        if (has_surface_elevations)
            assert (j < df.columns[surface_elevation_index].size ());
        if (has_bathy_elevations)
            assert (j < df.columns[bathy_elevation_index].size ());

        // Make assignments
        dataset[j].h5_index = df.columns[index_ph][j];
        dataset[j].x = df.columns[x_index][j];
        dataset[j].z = df.columns[z_index][j];
        if (has_manual_label)
            dataset[j].cls = df.columns[cls_index][j];
        if (has_predictions)
            dataset[j].prediction = df.columns[prediction_index][j];
        if (has_surface_elevations)
            dataset[j].surface_elevation = df.columns[surface_elevation_index][j];
        if (has_bathy_elevations)
            dataset[j].bathy_elevation = df.columns[bathy_elevation_index][j];
    }

    return dataset;
}

template<typename T>
std::vector<sample> read_training_samples (const bool verbose, const T &fns)
{
    using namespace std;
    using namespace ATL24_qtrees::dataframe;

    vector<sample> samples;

    // Read each file
    for (size_t i = 0; i < fns.size (); ++i)
    {
        if (verbose)
            clog << "Reading " << i << ": " << fns[i] << endl;

        const auto df = read (fns[i]);

        if (verbose)
        {
            clog << df.rows () << " rows read" << endl;
            clog << "Total photons = " << df.rows () << endl;
            clog << "Total dataframe columns = " << df.headers.size () << endl;
            // Print the columns headers
            // for (size_t j = 0; j < df.headers.size (); ++j)
            //     clog << "header[" << j << "]\t\'" << df.headers[j] << "'" << endl;
        }

        if (!df.has_column ("manual_label"))
            throw runtime_error ("Can't train without labelled data");

        // Convert them to the correct format
        auto tmp = convert_dataframe (df);

        // Set the ID
        for (auto &j : tmp)
            j.dataset_id = i;

        // Add the rows to the dataframe
        samples.insert (samples.end (), tmp.begin (), tmp.end ());
    }

    if (verbose)
    {
        clog << samples.size () << " samples read" << endl;
        unordered_map<size_t,size_t> label_map;
        for (const auto& i : samples)
            ++label_map[i.cls];
        clog << "label\ttotal\t%" << endl;
        clog << fixed;
        clog << setprecision (1);
        for (const auto& i : label_map)
        {
            clog << i.first;
            clog << "\t" << i.second;
            clog << "\t" << i.second * 100.0 / samples.size ();
            clog << endl;
        }
    }

    return samples;
}

template<typename T,typename U>
std::vector<size_t> get_sample_indexes (const T &samples,
    U &rng,
    const unsigned balance_priors_ratio)
{
    using namespace std;

    // Go through the samples in a random order
    vector<size_t> random_indexes (samples.size ());

    // 0, 1, 2, ...
    iota (random_indexes.begin (), random_indexes.end (), 0);

    shuffle (random_indexes.begin (), random_indexes.end (), rng);

    // '0' means don't do any prior balancing
    if (balance_priors_ratio == 0)
        return random_indexes;

    // Count occurrance of each label in each dataset
    unordered_map<size_t,unordered_map<size_t,size_t>> label_counts;
    for (auto i : samples)
        ++label_counts[i.dataset_id][i.cls];

    // Get number to assign for each label from each dataset
    unordered_map<size_t,size_t> max_samples;

    // Init
    for (auto i : label_counts)
        max_samples[i.first] = numeric_limits<size_t>::max();

    // Get minimum
    for (auto i : label_counts)
        for (auto j : i.second)
            max_samples[i.first] = std::min (j.second, max_samples[i.first]);

    // Get the sample indexes
    vector<size_t> sample_indexes;

    // Count current total from each dataset
    unordered_map<size_t,unordered_map<size_t,size_t>> sample_counts;

    for (size_t i = 0; i < samples.size (); ++i)
    {
        // Access in random order
        const size_t j = random_indexes[i];
        assert (j < samples.size ());
        const auto p = samples[j];

        // Apply 'balance_priors_ratio' to the counts.
        //
        // E.g., if balance_priors_ratio==3 then we want the ratio of
        // labels 0:40:41 (noise:bathy:surface) to be 3:1:3
        size_t max = max_samples[p.dataset_id];

        // If it's noise or surface, apply the specified ratio
        if (p.cls == 0 || p.cls == 41)
            max *= balance_priors_ratio;

        // Short circuit if we have already added enough points with this label
        if (sample_counts[p.dataset_id][p.cls] == max)
            continue;

        // Add it
        sample_indexes.push_back (j);

        // Count it
        ++sample_counts[p.dataset_id][p.cls];
    }

    return sample_indexes;
}

template<typename T,typename U,typename V>
void dump (const std::string &fn,
    const T &features,
    const size_t rows,
    const size_t cols,
    const U &labels,
    const V &dataset_ids)
{
    using namespace std;

    // Check invariants
    assert (!features.empty ());
    assert (features.size () == rows * cols);
    assert (labels.size () == rows);

    // Open the file for writing
    ofstream ofs (fn);

    if (!ofs)
        throw runtime_error ("Could not open file for writing");

    // Dump the column labels
    ofs << "label";
    ofs << ",dataset_id";
    for (size_t i = 0; i < cols; ++i)
        ofs << ',' << "f" << to_string(i);
    ofs << endl;

    // Dump the values to a CSV
    for (size_t i = 0; i < rows; ++i)
    {
        ofs << to_string (labels[i]);
        ofs << ',';
        ofs << to_string (dataset_ids[i]);
        for (size_t j = 0; j < cols; ++j)
        {
            ofs << ',';
            ofs << to_string (features[i * cols + j]);
        }
        ofs << endl;
    }
}

template<typename T>
size_t count_predictions (const T &p, const unsigned cls)
{
    return std::count_if (p.begin (), p.end (), [&](auto i)
    { return i.prediction == cls; });
}

// Get the average elevation in each one meter window along the track.
//
// Only average points with label 'cls'.
//
// Windows that contain no point with label 'cls' will contain NAN.
template<typename T>
std::vector<double> get_quantized_average (const T &p, const unsigned cls)
{
    using namespace std;

    assert (!p.empty ());

    // Get extent along the x axis
    const size_t min_x = min_element (p.begin (), p.end (),
        [](const auto &a, const auto &b)
        { return a.x < b.x; })->x;
    const size_t max_x = max_element (p.begin (), p.end (),
        [](const auto &a, const auto &b)
        { return a.x < b.x; })->x + 1.0;
    const size_t total = max_x - min_x;

    // Get 1m window averages
    vector<double> sums (total);
    vector<double> totals (total);
    for (size_t i = 0; i < p.size (); ++i)
    {
        // Skip non-'cls' photons
        if (p[i].prediction != cls)
            continue;

        // Get quantized along-track index
        const size_t j = p[i].x - min_x;

        // Check logic
        assert (j < totals.size ());
        assert (j < sums.size ());

        // Count the value
        ++totals[j];
        sums[j] += p[i].z;
    }

    // Get the average
    vector<double> avg (total, NAN);
    for (size_t i = 0; i < avg.size (); ++i)
        if (totals[i] != 0)
            avg[i] = sums[i] / totals[i];

    return avg;
}

// For each block of consecutive nan's, get the indexes of the two
// non-nan values on either side of the block
template<typename T>
std::vector<std::pair<size_t,size_t>> get_nan_pairs (const T &p)
{
    using namespace std;

    // Return value
    vector<pair<size_t,size_t>> np;

    // Special case for first value
    if (std::isnan (p[0]))
        np.push_back (make_pair (0, 0));

    // Get the left side of each block
    for (size_t i = 0; i + 1 < p.size (); ++i)
    {
        if (!std::isnan (p[i]) && isnan (p[i + 1]))
            np.push_back (make_pair (i, i));
    }

    // Find the right side of each block
    for (size_t i = 0; i < np.size (); ++i)
    {
        for (size_t j = np[i].first + 1; j < p.size (); ++j)
        {
            // Skip over nans
            if (std::isnan (p[j]))
                continue;

            // Set the index of the right side
            np[i].second = j;
            break;
        }
    }

    // Special case for last value
    if (std::isnan (p.back ()))
        np.back ().second = p.size () - 1;

    // Check our logic
    for (size_t i = 0; i < np.size (); ++i)
        assert (np[i].first < np[i].second);

    return np;
}

template<typename T,typename U>
void interpolate_nans (T &p, const U &n)
{
    using namespace std;

    // Check logic
    assert (n.first < n.second);

    // Get values to interpolate between
    double left = p[n.first];
    double right = p[n.second];

    // Check special cases
    if (std::isnan (left))
    {
        // First value is series was a NAN
        assert (n.first == 0);
        assert (!std::isnan (right));
        left = right;
        p[0] = right;
    }
    if (std::isnan (right))
    {
        // Last value in series was a nan
        assert (n.second == p.size () - 1);
        assert (!std::isnan (left));
        right = left;
        p[p.size () - 1] = left;
    }

    const double len = n.second - n.first;
    assert (len > 0.0);
    for (size_t i = n.first + 1; i < n.second; ++i)
    {
        // Distance from left
        const double d = i - n.first;
        assert (d > 0.0);
        assert (d < len);
        // Get weight from right
        const double w = d / len;
        assert (w > 0.0);
        assert (w < 1.0);
        // Interpolate
        const double avg = (1.0 - w) * left + w * right;
        assert (i < p.size ());
        p[i] = avg;
    }
}

template<typename T>
T box_filter (const T &p, const int filter_width)
{
    // Check invariants
    assert ((filter_width & 1) != 0); // Must be an odd kernel
    assert (filter_width >= 3); // w=1 does not make sense

    // Keep cumulative sums and totals across the vector
    std::vector<double> sums (p.size ());
    std::vector<size_t> totals (p.size ());
    double cumulative_sum = 0.0;
    size_t cumulative_total = 0;

    // For each pixel get cumulative sums and counts
    for (size_t i = 0; i < p.size (); ++i)
    {
        // Get running sum and total
        cumulative_sum += p[i];
        ++cumulative_total;

        // Remember them
        sums[i] = cumulative_sum;
        totals[i] = cumulative_total;
    }

    // Return value
    T q (p.size ());

    // Now go back and fill in filter values based upon sums and totals
    for (size_t i = 0; i < q.size (); ++i)
    {
        const int i1 = i - filter_width / 2 - 1;
        const int i2 = i + filter_width / 2;

        // Clip the index values to the edges of the row
        const double sum1 = (i1 < 0) ? 0 : sums[i1];
        const size_t total1 = (i1 < 0) ? 0 : totals[i1];
        const double sum2 = (i2 >= static_cast<int>(p.size ())) ? sums[p.size () - 1] : sums[i2];
        const size_t total2 = (i2 >= static_cast<int>(p.size ())) ? totals[p.size () - 1] : totals[i2];

        // Compute sum and total at pixel 'i'.
        const double sum = sum2 - sum1;
        const int total = total2 - total1;
        assert (total > 0);

        // Get the average over the window of size 'filter_width' at pixel 'i'
        q[i] = sum / total;
    }

    return q;
}

// Get elevation estimates for label 'cls' given a smoothing parameter 'sigma'.
template<typename T>
std::vector<double> get_elevation_estimates (const T &p, const double sigma, const unsigned cls)
{
    using namespace std;

    // Return value
    vector<double> z (p.size (), numeric_limits<double>::max ());

    // Count number of 'cls' predictions
    const size_t total = count_predictions (p, cls);

    // Degenerate case
    if (total == 0)
        return z;

    // Get 1m window averages
    auto avg = get_quantized_average (p, cls);

    // Get indexes of values to interpolate between
    const auto np = get_nan_pairs (avg);

    // Interpolate values between each pair
    for (auto n : np)
        interpolate_nans (avg, n);

    // All NANs should have been interpolated
    for (auto i : avg)
    {
        assert (!isnan (i));
        ((void) (i)); // Silence 'unused variable, i' warning
    }

    // Run a box filter over the averaged values
    const unsigned iterations = 4;

    // See: https://www.peterkovesi.com/papers/FastGaussianSmoothing.pdf
    const double ideal_filter_width = std::sqrt ((12.0 * sigma * sigma) / iterations + 1.0);
    const int filter_width = std::max (static_cast<int> (ideal_filter_width / 2.0), 1) * 2 + 1;

    // Apply Gaussian smoothing
    for (size_t i = 0; i < iterations; ++i)
        avg = box_filter (avg, filter_width);

    // Get min extent
    const double min_x = min_element (p.begin (), p.end (),
        [](const auto &a, const auto &b)
        { return a.x < b.x; })->x;

    // Fill in the estimates with the filtered points
    for (size_t i = 0; i < z.size (); ++i)
    {
        assert (i < p.size ());
        assert (min_x <= p[i].x);

        // Get along-track index
        const unsigned j = p[i].x - min_x;

        // Count the value
        assert (i < z.size ());
        assert (j < avg.size ());
        z[i] = avg[j];
    }

    return z;
}

template<typename T>
void assign_surface_estimates (T &samples, const double sigma)
{
    const auto e = get_elevation_estimates (samples, sigma, constants::sea_surface_class);

    assert (e.size () == samples.size ());

#pragma omp parallel for
    for (size_t i = 0; i < samples.size (); ++i)
        samples[i].surface_elevation = e[i];
}

template<typename T>
void assign_bathy_estimates (T &samples, const double sigma)
{
    const auto e = get_elevation_estimates (samples, sigma, constants::bathy_class);

    assert (e.size () == samples.size ());

#pragma omp parallel for
    for (size_t i = 0; i < samples.size (); ++i)
        samples[i].bathy_elevation = e[i];
}

template<typename T>
void check_surface_estimates (T &samples, size_t &changed)
{
    using namespace std;
    using namespace constants;

    // Keep track of how many changed using rollback semantics.
    size_t tmp = 0;

#pragma omp parallel for
    for (size_t i = 0; i < samples.size (); ++i)
    {
        // Skip non-surface photons
        if (samples[i].prediction != sea_surface_class)
            continue;

        // Surface photons must be in range
        if (samples[i].z < min_surface_elevation || samples[i].z > max_surface_elevation)
        {
            // Re-assign
            samples[i].prediction = 0;

#pragma omp atomic
            ++tmp;
            continue;
        }

        // Surface photons must be near the surface estimate
        const double delta = fabs (samples[i].z - samples[i].surface_elevation);

        if (delta > max_surface_estimate_delta)
        {
            // Re-assign
            samples[i].prediction = 0;

#pragma omp atomic
            ++tmp;
            continue;
        }
    }

    // Commit changes
    changed = tmp;
}

template<typename T>
void check_bathy_estimates (T &samples, size_t &changed)
{
    using namespace std;
    using namespace constants;

    // Keep track of how many changed using rollback semantics.
    size_t tmp = 0;

#pragma omp parallel for
    for (size_t i = 0; i < samples.size (); ++i)
    {
        // Skip non-bathy photons
        if (samples[i].prediction != bathy_class)
            continue;

        // Bathy photons must be in range
        if (samples[i].z < min_photon_elevation || samples[i].z > max_photon_elevation)
        {
            // Re-assign
            samples[i].prediction = 0;

#pragma omp atomic
            ++tmp;

            continue;
        }

        // Bathy photons must be deeper than the surface by some margin
        if (samples[i].z + min_bathy_depth >= samples[i].surface_elevation)
        {
            // Re-assign
            samples[i].prediction = 0;

#pragma omp atomic
            ++tmp;

            continue;
        }

        // Bathy photons must be near the bathy estimate
        const double delta = fabs (samples[i].z - samples[i].bathy_elevation);

        if (delta > max_bathy_estimate_delta)
        {
            // Re-assign
            samples[i].prediction = 0;

#pragma omp atomic
            ++tmp;

            continue;
        }
    }

    // Commit changes
    changed = tmp;
}

template<typename T,typename U>
void write_samples (std::ostream &os, T df, const U &samples)
{
    using namespace std;

    // Check invariants
    assert (df.rows () == samples.size ());

    // Pull data from samples
    vector<double> p (samples.size ());
    vector<double> s (samples.size ());
    vector<double> b (samples.size ());

#pragma omp parallel for
    for (size_t i = 0; i < samples.size (); ++i)
    {
        p[i] = samples[i].prediction;
        s[i] = samples[i].surface_elevation;
        b[i] = samples[i].bathy_elevation;
    }

    // Add new headers
    df.headers.push_back ("prediction");
    df.headers.push_back ("sea_surface_h");
    df.headers.push_back ("bathy_h");

    // Add new data
    df.columns.push_back (p);
    df.columns.push_back (s);
    df.columns.push_back (b);
    assert (df.is_valid ());

    // Write it out
    write (os, df);
}

class timer
{
private:
    std::chrono::time_point<std::chrono::system_clock> t1;
    std::chrono::time_point<std::chrono::system_clock> t2;
    bool running;

public:
    timer () : running (false)
    {
        start ();
    }
    void start ()
    {
        t1 = std::chrono::system_clock::now ();
        running = true;
    }
    void stop ()
    {
        t2 = std::chrono::system_clock::now ();
        running = false;
    }
    double elapsed_ms()
    {
        using namespace std::chrono;
        return running
            ? duration_cast<milliseconds> (system_clock::now () - t1).count ()
            : duration_cast<milliseconds> (t2 - t1).count ();
    }
};

} // namespace utils

} // namespace ATL24_qtrees

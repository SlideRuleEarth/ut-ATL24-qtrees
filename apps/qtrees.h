#pragma once

#include "precompiled.h"

namespace ATL24_qtrees
{

template<typename T,typename U>
std::vector<ATL24_qtrees::utils::sample> classify (const T &args, const U &photons)
{
    using namespace std;
    using namespace ATL24_qtrees::utils;
    using namespace ATL24_qtrees::utils::constants;
    using namespace ATL24_qtrees::xgboost;

    if (args.model_filename.empty ())
        throw runtime_error ("No model filename was specified");

    // Create the booster
    xgbooster xgb (args.verbose);
    xgb.load_model (args.model_filename);

    // Convert it to the correct format
    auto samples = convert_dataframe (photons);

    if (args.verbose)
    {
        clog << samples.size () << " samples read" << endl;
        clog << "Creating features" << endl;
    }

    // Save the photon indexes
    vector<size_t> h5_indexes (samples.size ());

#pragma omp parallel for
    for (size_t i = 0; i < h5_indexes.size (); ++i)
        h5_indexes[i] = samples[i].h5_index;

    feature_params fp;
    features f (samples, fp);

    // Create the raw data that gets passed to xgbooster
    const size_t rows = samples.size ();
    const size_t cols = f.features_per_sample ();
    vector<float> features; features.reserve (rows * cols);
    vector<uint32_t> labels; labels.reserve (rows);

    if (args.verbose)
        clog << "Features per sample " << f.features_per_sample () << endl;

    for (size_t i = 0; i < samples.size (); ++i)
    {
        // Get a row
        const auto row = f.get_features (i);

        // Append row to matrix
        features.insert (features.end (), row.begin (), row.end ());

        // Save the label
        labels.push_back (samples[i].cls);
    }

    // Check invariants
    assert (features.size () == rows * cols);
    assert (labels.size () == rows);

    // Get predictions
    {
        if (args.verbose)
            clog << "Getting predictions" << endl;

        const auto predictions = xgb.predict (features, rows, cols);

        if (args.verbose)
        {
            size_t correct = 0;

            assert (labels.size () == predictions.size ());
            for (size_t i = 0; i < labels.size (); ++i)
                correct += (labels[i] == predictions[i]);
            clog << fixed;
            clog << setprecision (1);
            clog << 100.0 * correct / predictions.size () << "% correct" << endl;
            clog << "Writing dataframe" << endl;
        }

        // Assign predictions
        assert (samples.size () == predictions.size ());
        for (size_t i = 0; i < samples.size (); ++i)
            samples[i].prediction = predictions[i];
    }

    // Check predictions in multiple passes
    const size_t passes = 2;

    // Assign estimates in-place
    assign_surface_estimates (samples, surface_sigma);

    // Check/reassign in multiple passes
    for (size_t pass = 0; pass < passes; ++pass)
    {
        // Check estimates in-place
        size_t changed = 0;
        check_surface_estimates (samples, changed);

        // Re-compute and assign estimates in case predictions changed
        assign_surface_estimates (samples, surface_sigma);
    }

    // Assign estimates in-place
    assign_bathy_estimates (samples, bathy_sigma);

    // Check/reassign in multiple passes
    for (size_t pass = 0; pass < passes; ++pass)
    {
        // Check estimates in-place
        size_t changed = 0;
        check_bathy_estimates (samples, changed);

        // Re-compute and assign estimates in case predictions changed
        assign_bathy_estimates (samples, bathy_sigma);
    }

    // Check invariants: The samples should be in the same order in which
    // they were read
#pragma omp parallel for
    for (size_t i = 0; i < samples.size (); ++i)
    {
        assert (h5_indexes[i] == samples[i].h5_index);
        ((void) (i)); // Eliminate unused variable warning
    }

    return samples;
}

} // namespace ATL24_qtrees

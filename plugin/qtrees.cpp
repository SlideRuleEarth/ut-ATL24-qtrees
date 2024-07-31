/******************************************************************************
 * INCLUDES
 ******************************************************************************/

#include "precompiled.h"
#include "classify_cmd.h"
#include "ATL24_qtrees/xgboost.h"

#include "icesat2/BathyFields.h"
#include "OsApi.h"
#include "EventLib.h"
#include "LuaEngine.h"

using BathyFields::extent_t;
using BathyFields::photon_t;

/******************************************************************************
 * DEFINES
 ******************************************************************************/

#define LUA_QTREES_LIBNAME    "qtrees"

/******************************************************************************
 * LOCAL FUNCTIONS
 ******************************************************************************/

/*----------------------------------------------------------------------------
 * qtrees_version
 *----------------------------------------------------------------------------*/
int qtrees_version (lua_State* L)
{
    lua_pushstring(L, BINID);
    lua_pushstring(L, BUILDINFO);
    return 2;
}

/*----------------------------------------------------------------------------
 * qtrees_open
 *----------------------------------------------------------------------------*/
int qtrees_open (lua_State *L)
{
    static const struct luaL_Reg qtrees_functions[] = {
        {"version",             qtrees_version},
        {NULL,                  NULL}
    };

    /* Set Library */
    luaL_newlib(L, qtrees_functions);

    return 1;
}

/*----------------------------------------------------------------------------
 * qtrees_classify
 *----------------------------------------------------------------------------*/
int qtrees_classify (std::vector<extent_t*> extents)
{
    using namespace std;
    using namespace ATL24_qtrees;
    using namespace ATL24_qtrees::dataframe;
    using namespace ATL24_qtrees::utils;
    using namespace ATL24_qtrees::utils::constants;
    using namespace ATL24_qtrees::xgboost;

    try
    {
        // Parameters
        bool verbose = true;
        string model_filename = "/data/model-20240607.json";

        // Create the booster
        xgbooster xgb (verbose);
        xgb.load_model (model_filename);

        // Get number of samples
        size_t number_of_samples = 0;
        for(size_t i = 0; i < extents.size(); i++)
        {
            number_of_samples += extents[i]->photon_count;
        }

        // Preallocate samples vector
        std::vector<utils::sample> samples;
        samples.reserve(number_of_samples);
        mlog(INFO, "Building %ld photon samples", number_of_samples);

        // Build and add samples
        for(size_t i = 0; i < extents.size(); i++)
        {
            for(size_t j = 0; j < extents[i]->photon_count; j++)
            {
                photon_t* photons = extents[i]->photons;
                utils::sample s = {
                    .h5_index = static_cast<size_t>(photons[j].index_ph),
                    .x = photons[j].x_atc,
                    .z = photons[j].ortho_h
                };
                samples.push_back(s);
            }
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
            const auto predictions = xgb.predict (features, rows, cols);

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

        // Save results
//        write_samples (cout, photons, samples);

        return 0;
    }
    catch (const exception &e)
    {
        cerr << e.what () << endl;
        return -1;
    }
}

/******************************************************************************
 * EXPORTED FUNCTIONS
 ******************************************************************************/

extern "C" {
void initqtrees (void)
{
    /* Extend Lua */
    LuaEngine::extend(LUA_QTREES_LIBNAME, qtrees_open);

    /* Indicate Presence of Package */
    LuaEngine::indicate(LUA_QTREES_LIBNAME, BINID);

    /* Display Status */
    print2term("%s plugin initialized (%s)\n", LUA_QTREES_LIBNAME, BINID);
}

void deinitqtrees (void)
{
}
}

/******************************************************************************
 * INCLUDES
 ******************************************************************************/

#include "precompiled.h"
#include "classify_cmd.h"
#include "ATL24_qtrees/xgboost.h"

#include "QtreesClassifier.h"
#include "bathy/BathyParms.h"

/******************************************************************************
 * EXTERNAL FUNCTION (to be moved)
 ******************************************************************************/

using namespace std;
using namespace ATL24_qtrees;
using namespace ATL24_qtrees::dataframe;
using namespace ATL24_qtrees::utils;
using namespace ATL24_qtrees::utils::constants;
using namespace ATL24_qtrees::xgboost;

void classify (bool verbose, string model_filename, vector<utils::sample>& samples)
{
    // Create the booster
    xgbooster xgb (verbose);
    xgb.load_model (model_filename);

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

    if (verbose)
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
        if (verbose)
            clog << "Getting predictions" << endl;

        const auto predictions = xgb.predict (features, rows, cols);

        if (verbose)
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
}

/******************************************************************************
 * DATA
 ******************************************************************************/

const char* QtreesClassifier::CLASSIFIER_NAME = "qtrees";
const char* QtreesClassifier::QTREES_PARMS = "qtrees";
const char* QtreesClassifier::DEFAULT_QTREES_MODEL = "/data/model-20240607.json";

static const char* QTREES_PARM_MODEL = "model";
static const char* QTREES_PARM_SET_CLASS = "set_class";
static const char* QTREES_PARM_SET_SURFACE = "set_surface";
static const char* QTREES_PARM_VERBOSE = "verbose";

/******************************************************************************
 * METHODS
 ******************************************************************************/

/*----------------------------------------------------------------------------
 * luaCreate - create(parms)
 *----------------------------------------------------------------------------*/
int QtreesClassifier::luaCreate (lua_State* L)
{
    try
    {
        return createLuaObject(L, new QtreesClassifier(L, 1));
    }
    catch(const RunTimeException& e)
    {
        mlog(e.level(), "Error creating QtreesClassifier: %s", e.what());
        return returnLuaStatus(L, false);
    }
}

/*----------------------------------------------------------------------------
 * init
 *----------------------------------------------------------------------------*/
void QtreesClassifier::init (void)
{
}

/*----------------------------------------------------------------------------
 * Constructor
 *----------------------------------------------------------------------------*/
QtreesClassifier::QtreesClassifier (lua_State* L, int index):
    BathyClassifier(L, CLASSIFIER_NAME)
{
    /* Build Parameters */
    if(lua_istable(L, index))
    {
        /* model */
        lua_getfield(L, index, QTREES_PARM_MODEL);
        parms.model = LuaObject::getLuaString(L, -1, true, parms.model.c_str());
        lua_pop(L, 1);

        /* set class */
        lua_getfield(L, index, QTREES_PARM_SET_CLASS);
        parms.set_class = LuaObject::getLuaBoolean(L, -1, true, parms.set_class);
        lua_pop(L, 1);

        /* set surface */
        lua_getfield(L, index, QTREES_PARM_SET_SURFACE);
        parms.set_surface = LuaObject::getLuaBoolean(L, -1, true, parms.set_surface);
        lua_pop(L, 1);

        /* verbose */
        lua_getfield(L, index, QTREES_PARM_VERBOSE);
        parms.verbose = LuaObject::getLuaBoolean(L, -1, true, parms.verbose);
        lua_pop(L, 1);
    }
}

/*----------------------------------------------------------------------------
 * Destructor
 *----------------------------------------------------------------------------*/
QtreesClassifier::~QtreesClassifier (void)
{
}

/*----------------------------------------------------------------------------
 * run
 *----------------------------------------------------------------------------*/
bool QtreesClassifier::run (const vector<BathyParms::extent_t*>& extents)
{
    try
    {
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
            BathyParms::photon_t* photons = extents[i]->photons;
            for(size_t j = 0; j < extents[i]->photon_count; j++)
            {
                utils::sample s = {
                    .h5_index = static_cast<size_t>(photons[j].index_ph),
                    .x = photons[j].x_atc,
                    .z = photons[j].ortho_h
                };
                samples.push_back(s);
            }
        }

        // Run classification
        classify(parms.verbose, parms.model, samples);

        // Update extents
        size_t s = 0; // sample index
        for(size_t i = 0; i < extents.size(); i++)
        {
            BathyParms::photon_t* photons = extents[i]->photons;
            for(size_t j = 0; j < extents[i]->photon_count; j++)
            {
                if(parms.set_surface) photons[j].surface_h = samples[s].surface_elevation;
                if(parms.set_class) photons[j].class_ph = samples[s].prediction;
                photons[j].predictions[classifier] = samples[s].prediction;
                s++; // go to next sample
            }
        }
    }
    catch (const std::exception &e)
    {
        mlog(CRITICAL, "Failed to run qtrees classifier: %s", e.what());
        return false;
    }

    return true;
}

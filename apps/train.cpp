#include "precompiled.h"
#include "train_cmd.h"
#include "ATL24_qtrees/xgboost.h"

using namespace std;
using namespace ATL24_qtrees;
using namespace ATL24_qtrees::utils;
using namespace ATL24_qtrees::xgboost;

const string usage {"ls *.csv | train [options]"};

// Train an XGBoost model given the samples in 'samples'
template<typename T,typename U,typename V>
xgbooster train (const T &args,
    const U &samples,
    const V &fp,
    double &accuracy,
    bool predict)
{
    if (args.verbose)
        clog << "Creating features" << endl;

    const features f (samples, fp);

    // Create the random number generator
    mt19937_64 rng (args.random_seed);

    if (args.verbose)
        clog << "Getting sample indexes" << endl;

    const vector<size_t> sample_indexes = get_sample_indexes (samples,
        rng,
        args.balance_priors_ratio);

    if (args.verbose)
    {
        clog << "Training with " << sample_indexes.size () << " total samples" << endl;
        unordered_map<size_t,size_t> label_map;
        for (auto i : sample_indexes)
            ++label_map[samples[i].cls];
        clog << "label\ttotal\t%" << endl;
        clog << fixed;
        clog << setprecision (1);
        for (auto i : label_map)
        {
            clog << i.first;
            clog << "\t" << i.second;
            clog << "\t" << i.second * 100.0 / sample_indexes.size ();
            clog << endl;
        }
        clog << "Creating training data" << endl;
    }

    // Create the raw data that gets passed to xgbooster
    const size_t rows = sample_indexes.size ();
    const size_t cols = f.features_per_sample ();
    vector<float> features; features.reserve (rows * cols);
    vector<uint32_t> labels; labels.reserve (rows);
    vector<uint32_t> dataset_ids; dataset_ids.reserve (rows);

    if (args.verbose)
        clog << "Features per sample " << f.features_per_sample () << endl;

    for (size_t i = 0; i < sample_indexes.size (); ++i)
    {
        // Get a row
        const auto j = sample_indexes[i];
        const auto row = f.get_features (j);

        // Append row to matrix
        features.insert (features.end (), row.begin (), row.end ());

        // Save label to vector
        const uint32_t label = remap_label (samples[j].cls);
        labels.push_back (label);

        // Save dataset ID to vector
        const uint32_t id = samples[j].dataset_id;
        dataset_ids.push_back (id);
    }

    // Dump features, if specified
    if (!args.feature_dump_filename.empty ())
    {
        if (args.verbose)
            clog << "Dumping features to " << args.feature_dump_filename << endl;

        dump (args.feature_dump_filename, features, rows, cols, labels, dataset_ids);
    }

    // Check invariants
    assert (features.size () == rows * cols);
    assert (labels.size () == rows);

    // Create the booster
    xgbooster xgb (args.verbose);

    // Warm start if an input model was specified
    if (!args.input_model_filename.empty ())
        xgb.load_model (args.input_model_filename);

    // Train the model
    xgb.train (features, labels, rows, cols, args.epochs);

    if (predict)
    {
        const auto predictions = xgb.predict (features, rows, cols);

        double total_correct = 0;

        assert (labels.size () == predictions.size ());
        for (size_t i = 0; i < labels.size (); ++i)
            total_correct += (unremap_label (labels[i]) == predictions[i]);

        accuracy = total_correct / predictions.size ();

        if (args.verbose)
        {
            clog << "Feature parameters" << endl;
            clog << fp;
            clog << "Training accuracy = " << accuracy << endl;
        }
    }

    return xgb;
}

// train() overload
template<typename T,typename U,typename V>
xgbooster train (const T &args, const U &samples, const V &fp)
{
    double accuracy;
    bool predict = false;
    return train (args, samples, fp, accuracy, predict);
}

// train() overload
template<typename T,typename U,typename V>
xgbooster train (const T &args, const U &samples, const V &fp, double &accuracy)
{
    bool predict = true;
    return train (args, samples, fp, accuracy, predict);
}

int main (int argc, char **argv)
{
    try
    {
        // Parse the args
        const auto args = cmd::get_args (argc, argv, usage);

        // If you are getting help, exit without an error
        if (args.help)
            return 0;

        if (args.verbose)
        {
            // Show the args
            clog << "cmd_line_parameters:" << endl;
            clog << args << endl;
        }

        if (!args.search && args.output_model_filename.empty ())
            throw runtime_error ("No output model filename was specified");

        // Read input filenames
        vector<string> fns;

        if (args.verbose)
            clog << "Reading filenames from stdin" << endl;

        // Read filenames from stdin
        for (string line; getline(cin, line); )
            fns.push_back (line);

        if (args.verbose)
            clog << fns.size () << " filenames read" << endl;

        // Read all of the data
        const auto samples = read_training_samples (args.verbose, fns);

        if (args.search)
        {
            vector<feature_params> fps;
            for (auto window_size : {30.0, 40.0, 50.0})
                for (auto total_quantiles : {32u, 48u, 64u, 80u, 96u})
                    for (auto adjacent_windows : {2u, 3u, 4u})
                        fps.push_back (feature_params {window_size, total_quantiles, adjacent_windows});

            // Compute accuracy for each set of parameters
            vector<double> accuracies;
            for (auto fp : fps)
            {
                // Get the trained model
                double accuracy;
                const auto xgb = train (args, samples, fp, accuracy);

                if (args.verbose)
                    clog << "accuracy = " << accuracy << endl;

                accuracies.push_back (accuracy);
            }

            // Choose the parameters that yield the best accuracy
            assert (accuracies.size () == fps.size ());

            // Dump in table format
            clog << "acc\tws\ttq\taw" << endl;
            for (size_t i = 0; i < accuracies.size (); ++i)
            {
                clog << accuracies[i];
                clog << "\t" << fps[i].window_size;
                clog << "\t" << fps[i].total_quantiles;
                clog << "\t" << fps[i].adjacent_windows;
                clog << endl;
            }

            // Get the best parameters
            double best_accuracy = 0.0;
            auto best_fp = fps[0];
            for (size_t i = 0; i < accuracies.size (); ++i)
            {
                if (accuracies[i] < best_accuracy)
                    continue;

                best_accuracy = accuracies[i];
                best_fp = fps[i];
            }

            if (args.verbose)
            {
                // Show the best
                clog << "Best accuracy = " << best_accuracy << endl;
                clog << "Best feature parameters" << endl;
                clog << best_fp << endl;
            }
        }
        else
        {
            // Get the trained model
            const feature_params fp;
            const auto xgb = train (args, samples, fp);

            // Save it
            xgb.save_model (args.output_model_filename);
        }

        return 0;
    }
    catch (const exception &e)
    {
        cerr << e.what () << endl;
        return -1;
    }
}

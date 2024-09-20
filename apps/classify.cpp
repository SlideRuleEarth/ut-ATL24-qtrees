#include "precompiled.h"
#include "classify_cmd.h"
#include "ATL24_qtrees/xgboost.h"
#include "qtrees.h"

const std::string usage {"classify [options] < input_filename.csv > output_filename.csv"};

int main (int argc, char **argv)
{
    using namespace std;
    using namespace ATL24_qtrees;
    using namespace ATL24_qtrees::dataframe;
    using namespace ATL24_qtrees::utils;

    try
    {
        // Keep track of performance
        timer total_timer;
        timer processing_timer;

        total_timer.start ();

        // Parse the args
        const auto args = cmd::get_args (argc, argv, usage);

        if (args.verbose)
        {
            clog << "cmd_line_parameters:" << endl;
            clog << args;
        }

        // If you are getting help, exit without an error
        if (args.help)
            return 0;

        // Read the input file
        if (args.verbose)
            clog << "Reading CSV from stdin" << endl;

        const auto photons = read (cin);

        if (args.verbose)
        {
            clog << "Total photons = " << photons.rows () << endl;
            clog << "Total dataframe columns = " << photons.headers.size () << endl;
        }

        processing_timer.start ();

        // Convert it to the correct format
        auto samples = convert_dataframe (photons);

        // Get the predictions
        classify (args, samples);

        processing_timer.stop ();

        // Save results
        write_samples (cout, photons, samples);

        total_timer.stop ();

        if (args.verbose)
        {
            clog << "Total elapsed time " << total_timer.elapsed_ms () / 1000.0 << " seconds" << endl;
            clog << "Elapsed processing time " << processing_timer.elapsed_ms () / 1000.0 << " seconds" << endl;
            clog << photons.rows () / (total_timer.elapsed_ms () / 1000.0) << " photons/second total" << endl;
            clog << photons.rows () / (processing_timer.elapsed_ms () / 1000.0) << " photons/second without I/O" << endl;
        }

        return 0;
    }
    catch (const exception &e)
    {
        cerr << e.what () << endl;
        return -1;
    }
}

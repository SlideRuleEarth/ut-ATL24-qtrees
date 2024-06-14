#pragma once

#include "precompiled.h"
#include "ATL24_qtrees/cmd_utils.h"

namespace ATL24_qtrees
{

namespace cmd
{

struct args
{
    bool help = false;
    bool verbose = false;
    unsigned balance_priors_ratio = 0;
    size_t random_seed = 123;
    size_t epochs = 100;
    bool search = false;
    std::string feature_dump_filename;
    std::string input_model_filename;
    std::string output_model_filename = std::string ("./model.json");
};

std::ostream &operator<< (std::ostream &os, const args &args)
{
    os << std::boolalpha;
    os << "help: " << args.help << std::endl;
    os << "verbose: " << args.verbose << std::endl;
    os << "balance-priors-ratio: " << args.balance_priors_ratio << std::endl;
    os << "random-seed: " << args.random_seed << std::endl;
    os << "epochs: " << args.epochs << std::endl;
    os << "search: " << args.search << std::endl;
    os << "feature-dump-filename: " << args.feature_dump_filename << std::endl;
    os << "input-model-filename: " << args.input_model_filename << std::endl;
    os << "output-model-filename: " << args.output_model_filename << std::endl;
    return os;
}

args get_args (int argc, char **argv, const std::string &usage)
{
    args args;
    while (1)
    {
        int option_index = 0;
        static struct option long_options[] = {
            {"help", no_argument, 0,  'h' },
            {"verbose", no_argument, 0,  'v' },
            {"balance-priors-ratio", required_argument, 0,  'b' },
            {"random-seed", required_argument, 0,  's' },
            {"epochs", required_argument, 0,  'e' },
            {"search", no_argument, 0,  'a' },
            {"feature-dump-filename", required_argument, 0,  'd' },
            {"input-model-filename", required_argument, 0,  'i' },
            {"output-model-filename", required_argument, 0,  'o' },
            {0,      0,           0,  0 }
        };

        int c = getopt_long(argc, argv, "hvb:s:e:ad:i:o:", long_options, &option_index);
        if (c == -1)
            break;

        switch (c) {
            default:
            case 0:
            case 'h':
            {
                const size_t noptions = sizeof (long_options) / sizeof (struct option);
                ATL24_utils::cmd::print_help (std::clog, usage, noptions, long_options);
                if (c != 'h')
                    throw std::runtime_error ("Invalid option");
                args.help = true;
                return args;
            }
            case 'v': args.verbose = true; break;
            case 'b': args.balance_priors_ratio = atol(optarg); break;
            case 's': args.random_seed = atol(optarg); break;
            case 'e': args.epochs = atol(optarg); break;
            case 'a': args.search = true; break;
            case 'd': args.feature_dump_filename = std::string(optarg); break;
            case 'i': args.input_model_filename = std::string(optarg); break;
            case 'o': args.output_model_filename = std::string(optarg); break;
        }
    }

    // Check command line
    if (optind != argc)
        throw std::runtime_error ("Too many command line parameters");

    return args;
}

} // namespace cmd

} // namespace ATL24_qtrees

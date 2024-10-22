#include "precompiled.h"
#include "ATL24_qtrees/utils.h"
#include "ATL24_qtrees/verify.h"

using namespace std;
using namespace ATL24_qtrees::utils;

struct tmp
{
    double x;
};

void test_get_window_indexes ()
{
    // Create some samples
    const size_t n = 11;
    double x = 0;
    vector<tmp> s (n);
    for (auto &j : s)
        j.x = x++;

    const double window_size = 10;
    const auto w = get_window_indexes (s, window_size);

    VERIFY (w[0] == 0);
    VERIFY (w[1] == 0);
    VERIFY (w[n - 2] == 0);
    VERIFY (w[n - 1] == 1);
}

int main ()
{
    try
    {
        test_get_window_indexes ();

        return 0;
    }
    catch (const exception &e)
    {
        cerr << e.what () << endl;
        return -1;
    }
}

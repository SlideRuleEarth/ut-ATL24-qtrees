#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace ATL24_qtrees
{

namespace dataframe
{

struct dataframe
{
    std::vector<std::string> headers;
    std::vector<std::vector<double>> columns;
    bool is_valid () const
    {
        // Number of headers match number of columns
        if (headers.size () != columns.size ())
            return false;
        // Number of rows are the same in each column
        for (size_t i = 1; i < columns.size (); ++i)
            if (columns[i].size () != columns[0].size ())
                return false;
        return true;
    }
    size_t rows () const
    {
        assert (is_valid ());
        return columns.empty () ? 0 : columns[0].size ();
    }
    // Check if a column name exists
    bool has_column (const std::string &name) const
    {
        using namespace std;
        const auto it = find (headers.begin (), headers.end (), name);
        return !(it == headers.end ());
    }
    // Access the column with header named 'h'
    const std::vector<double> &operator[] (const std::string &h) const
    {
        using namespace std;
        const auto it = find (headers.begin (), headers.end (), h);
        if (it == headers.end ())
            throw runtime_error (string ("Can't find dataframe column: ") + h);
        const size_t index = it - headers.begin ();
        return columns[index];
    }
    // Add row 'n' from the dataframe 'df' into this dataframe
    void add_row (const dataframe &df, const size_t n)
    {
        // Check invariants
        assert (df.is_valid ());
        assert (n < df.rows ());

        // Add the row
        for (size_t i = 0; i < columns.size (); ++i)
        {
            assert (i < columns.size ());
            assert (n < df.columns[i].size ());
            columns[i].push_back (df.columns[i][n]);
        }

        // Check invariants
        assert (is_valid ());
    }
    // Append all rows from 'df' to this dataframe
    void append (const dataframe &df)
    {
        // Check invariants
        assert (df.is_valid ());

        if (df.headers.size () != headers.size ())
            throw std::runtime_error ("The number of headers in the dataframes do not match");

        for (size_t i = 0; i < headers.size (); ++i)
            if (df.headers[i] != headers[i])
                throw std::runtime_error ("The header names do not match");

        // This has to be true or 'df' is not valid
        assert (df.columns.size () == columns.size ());

        for (size_t i = 0; i < columns.size (); ++i)
            columns[i].insert (columns[i].end (),
                df.columns[i].begin (),
                df.columns[i].end ());

        // Check invariants
        assert (is_valid ());
    }
};

dataframe read (std::istream &is)
{
    using namespace std;

    // Create the dataframe
    dataframe df;

    // Read the headers
    string line;

    if (!getline (is, line))
        return df;

    // Parse each individual column header
    stringstream ss (line);
    string header;
    while (getline (ss, header, ','))
    {
        // Remove LFs in case the file was created under Windows
        std::erase (header, '\r');

        // Save it
        df.headers.push_back (header);
    }

    // Allocate column vectors
    df.columns.resize (df.headers.size ());

    // Now get the rows
    while (getline (is, line))
    {
        // Skip empty lines
        if (line.empty ())
            continue;
        char *p = &line[0];
        for (size_t i = 0; i < df.headers.size (); ++i)
        {
            char *end;
            const double x = strtod (p, &end);
            df.columns[i].push_back (x);
            p = end;
            // Ignore ','
            if (*p == ',')
                ++p;
        }
    }

    assert (df.is_valid ());
    return df;
}

dataframe read (const std::string &fn)
{
    using namespace std;

    ifstream ifs (fn);
    if (!ifs)
        throw runtime_error ("Could not open file for reading");
    return ATL24_qtrees::dataframe::read (ifs);
}

std::ostream &write (std::ostream &os, const dataframe &df, const size_t precision = 16)
{
    using namespace std;

    assert (df.is_valid ());

    const size_t ncols = df.headers.size ();

    // Short-circuit
    if (ncols == 0)
        return os;

    // Print headers
    bool first = true;
    for (auto h : df.headers)
    {
        if (!first)
            os << ",";
        first = false;
        os << h;
    }
    os << endl;

    const size_t nrows = df.columns[0].size ();

    // Short-circuit
    if (nrows == 0)
        return os;

    // Save format
    const auto f = os.flags();
    const auto p = os.precision();

    // Set format
    os << std::fixed;
    os << std::setprecision (precision);

    // Write it out
    for (size_t i = 0; i < nrows; ++i)
    {
        for (size_t j = 0; j < ncols; ++j)
        {
            if (j != 0)
                os << ",";
            os << df.columns[j][i];
        }
        os << endl;
    }

    // Restore format
    os.precision (p);
    os.flags (f);

    return os;
}

std::ostream &operator<< (std::ostream &os, const dataframe &df)
{
    return write (os , df);
}

}

} // namespace ATL24_qtrees

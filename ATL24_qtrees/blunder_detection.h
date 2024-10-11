#pragma once

#include "precompiled.h"

namespace ATL24_qtrees
{

// ASPRS Definitions
constexpr unsigned bathy_class = 40;
constexpr unsigned sea_surface_class = 41;

namespace detail
{

template<typename T>
std::vector<size_t> get_nearest_along_track_prediction (const T &p, const unsigned c)
{
    // At each point in 'p', what is the index of the closest point
    // with the label 'c'?
    using namespace std;

    // Set sentinels
    vector<size_t> indexes (p.size (), p.size ());

    // Check data
    if (p.empty ())
        return indexes;

    // Get first and last indexes with label 'c'
    size_t first_index = p.size ();
    size_t last_index = p.size ();
    for (size_t i = 0; i < p.size (); ++i)
    {
        // Ignore ones that are the wrong class
        if (p[i].prediction != c)
            continue;

        // Only set it if it's the first
        if (first_index == p.size ())
            first_index = i;

        // Always set the last
        last_index = i;
    }

    // If we didn't find at least one, there is nothing to do
    if (first_index == p.size ())
        return indexes;

    // Check logic
    assert (last_index != p.size ());

    // Set all of the indexes to the left of 'first_index'
    for (size_t i = 0; i < first_index; ++i)
        indexes[i] = first_index;

    // Set all of the indexes to the right of 'last_index'
    for (size_t i = last_index; i < p.size (); ++i)
        indexes[i] = last_index;

    // Set sentinels
    size_t left_index = p.size ();
    size_t right_index = p.size ();

    // Now set all of the indexes between 'first_index' and
    // 'last_index'
    for (size_t i = first_index; i < last_index; ++i)
    {
        // Is this a label that we are interested in?
        if (p[i].prediction == c)
        {
            // Closest point with label 'c' is itself
            indexes[i] = i;

            // Save its position
            right_index = left_index = i;

            continue;
        }

        // Search to the right for the next index with label 'c'
        if (right_index < i)
        {
            for (size_t j = i; j <= last_index; ++j)
            {
                if (p[j].prediction == c)
                {
                    right_index = j;
                    break;
                }
            }
        }

        // Logic check
        assert (left_index < i);
        assert (i < right_index);
        assert (p[left_index].x <= p[i].x);
        assert (p[i].x <= p[right_index].x);

        // Set the index of the closer of the two
        const double d_left = p[i].x - p[left_index].x;
        const double d_right = p[right_index].x - p[i].x;

        if (d_left <= d_right)
            indexes[i] = left_index;
        else
            indexes[i] = right_index;
    }

    // Logic check
    for (size_t i = 0; i < indexes.size (); ++i)
        assert (indexes[i] < p.size ());

    return indexes;
}

template<typename T>
T surface_elevation_check (T p,
    const double surface_min_elevation,
    const double surface_max_elevation)
{
    assert (!p.empty ());

    // Surface photons must be near sea level
    for (size_t i = 0; i < p.size (); ++i)
    {
        // Ignore non-surface
        if (p[i].prediction != sea_surface_class)
            continue;

        // Too high?
        if (p[i].z > surface_max_elevation)
            p[i].prediction = 0;

        // Too low?
        if (p[i].z < surface_min_elevation)
            p[i].prediction = 0;
    }

    return p;
}

template<typename T>
T bathy_elevation_check (T p,
    const double bathy_min_elevation)
{
    assert (!p.empty ());

    // Bathy photons can't be too deep
    for (size_t i = 0; i < p.size (); ++i)
    {
        // Ignore non-bathy
        if (p[i].prediction != bathy_class)
            continue;

        // Too deep?
        if (p[i].z < bathy_min_elevation)
            p[i].prediction = 0;
    }

    return p;
}

template<typename T>
T relative_depth_check (T p, const double water_column_width)
{
    assert (!p.empty ());

    // Count surface photons
    const size_t total_surface = count_predictions (p, sea_surface_class);

    // If there is no surface, there is nothing to do
    if (total_surface == 0)
        return p;

    // Count bathy photons
    const size_t total_bathy = count_predictions (p, bathy_class);

    // If there is no bathy, there is nothing to do
    if (total_bathy == 0)
        return p;

    // We need to know the along-track distance to surface photons
    const auto nearby_surface_indexes = get_nearest_along_track_prediction (p, sea_surface_class);

    for (size_t i = 0; i < p.size (); ++i)
    {
        // Ignore non-bathy
        if (p[i].prediction != bathy_class)
            continue;

        // Get the closest surface index
        const size_t j = nearby_surface_indexes[i];

        // How far away is it?
        assert (j < p.size ());
        const double dx = fabs (p[i].x - p[j].x);

        // If it's too far away, we can't do the check
        if (dx > water_column_width)
            continue;

        // If the bathy below the surface?
        if (p[i].z < p[j].surface_elevation)
            continue; // Yes, keep going

        // No, reassign
        p[i].prediction = 0;
    }

    return p;
}

template<typename T>
T surface_range_check (T p, const double range)
{
    assert (!p.empty ());

    // Count surface photons
    const size_t total_surface = count_predictions (p, sea_surface_class);

    // If there is no surface, there is nothing to do
    if (total_surface == 0)
        return p;

    // Surface photons must be near the surface estimate
    for (size_t i = 0; i < p.size (); ++i)
    {
        // Ignore non-surface
        if (p[i].prediction != sea_surface_class)
            continue;

        // Is it close enough?
        const double d = std::fabs (p[i].z - p[i].surface_elevation);

        // Must be within +-range
        if (d > range)
            p[i].prediction = 0;
    }

    return p;
}

template<typename T>
T bathy_range_check (T p, const double range)
{
    assert (!p.empty ());

    // Count bathy photons
    const size_t total_bathy = count_predictions (p, bathy_class);

    // If there is no bathy, there is nothing to do
    if (total_bathy == 0)
        return p;

    // Bathy photons must be near the bathy estimate
    for (size_t i = 0; i < p.size (); ++i)
    {
        // Ignore non-bathy
        if (p[i].prediction != bathy_class)
            continue;

        // Is it close enough?
        const double d = std::fabs (p[i].z - p[i].bathy_elevation);

        // Must be within +-range
        if (d > range)
            p[i].prediction = 0;
    }

    return p;
}

} // namespace detail

template<typename T,typename U>
T blunder_detection (T p, const U &params)
{
    // Reclassify photons using heuristics
    using namespace std;

    if (p.empty ())
        return p;

    // Surface photons must be near sea level
    p = detail::surface_elevation_check (p,
        params.surface_min_elevation,
        params.surface_max_elevation);

    // Bathy photons can't be too deep
    p = detail::bathy_elevation_check (p,
        params.bathy_min_elevation);

    // Bathy photons can't be above the sea surface
    p = detail::relative_depth_check (p, params.water_column_width);

    // Sea surface photons must all be near the elevation estimate
    p = detail::surface_range_check (p, params.surface_range);

    // Bathy photons must all be near the elevation estimate
    p = detail::bathy_range_check (p, params.bathy_range);

    return p;
}

} // namespace ATL24_qtrees

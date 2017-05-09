#ifndef __COMMON_H__
#define __COMMON_H__
// #########################################
// Common header which defines some common
// classes and types in the project.
// #########################################
#include <vector>

// Descriptor for interest points
struct Descriptor {
    int x; // x-coor. in full scale
    int y; // y-coor. in full scale
    float fea[64];
};

typedef vector<Descriptor> FeatPts;

// Matching descriptors
struct MatchPts {
    Descriptor* dsp1;
    Descriptor* dsp2;
};

#endif

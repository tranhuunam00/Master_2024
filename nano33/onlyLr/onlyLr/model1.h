#ifndef MODEL1_H
#define MODEL1_H

#define NUM_FEATURES_1 6 // chỉ 6 features của model1

// logistic regression weights (1 vector)
const float weights1[NUM_FEATURES_1] = {
    -7.30782, -0.53781, -3.26122, -6.36183, 6.71940, 2.00442};

// bias (1 giá trị)
const float bias1 = -3.13008;

// min-max để scale 6 features
const float min_vals1[NUM_FEATURES_1] = {
    0.85574, 0.00098, -0.31000, -0.20100, -1.36000, -0.95000};

const float max_vals1[NUM_FEATURES_1] = {
    1.22997, 1.36570, 0.97000, 1.04800, 0.77700, 0.82500};
const float scale1[NUM_FEATURES_1] = {
    5.34428, 1.46550, 1.56250, 1.60128, 0.93589, 1.12676};
#endif

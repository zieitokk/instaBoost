#ifndef GUIDED_POS_H
#define GUIDED_POS_H

#include "heatmap.h"

/*
 *  In this function we try to do:
 *  1. generate heatmap_guided_position
 */

cv::Point _get_paste_pos(cv::Mat heatmap, float ratio);


#endif
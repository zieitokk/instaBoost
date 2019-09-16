#ifndef AFFINE_TRANSFORM_H
#define AFFINE_TRANSFORM_H

#include "single_img_process.h"
#include "guided_pos.h"

// Here we transform not only the target images and also its corresponding annotations(bbox, segmentation) after we randomly generated the guided_pos
void _transform(cv::Mat *target, Mask *an, cv::Mat *new_img, cv::Mat background, std::unordered_map<std::string, float> t, cv::Point guided_pos);

#endif
#ifndef HEATMAP_H
#define HEATMAP_H

#include "coco_mask_extractor.h"
#include <opencv2/opencv.hpp>

/*
 *  Here we have obtained the masks of instance, which are annotated by consecutive points.
 *  Next fill the mask with binary map, where 255 represents the background and 0 the instance.
 *
 */

cv::Mat _get_coco_masks(Mask mask_anno, imgInfo imginfo);

cv::Mat _get_trimap(cv::Mat mask, int kernel_size);

std::deque<std::vector<std::vector<int>>> _get_trimap_coord(cv::Mat trimap);

cv::Mat _get_mask_center_background(Mask mask_anno, imgInfo imginfo, cv::Mat img);

std::deque<std::vector<std::vector<int>>> _get_rings(std::deque<std::vector<std::vector<int>>> trimap_coord, cv::Mat img);

std::deque<std::vector<std::vector<int>>> _translate_trimap(std::deque<std::vector<std::vector<int>>> trimap_coord, cv::Point oripos, cv::Point augpos);

float _get_euclidean_distance(std::vector<std::vector<int>> oriRing, std::vector<std::vector<int>> augRing);

float _get_heat_point(std::deque<std::vector<std::vector<int>>> ori_trimap_coord, std::deque<std::vector<std::vector<int>>> oriRings, cv::Mat background, cv::Point oripos, cv::Point augpos, std::vector<float> config);

cv::Mat _get_heatmap(cv::Mat img, cv::Mat background, cv::Mat trimap, cv::Point center, int shrink);

cv::Mat _normalize(cv::Mat heatmap);

#endif
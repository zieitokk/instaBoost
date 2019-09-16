#ifndef SINGLE_IMG_PROCESS_H
#define SINGLE_IMG_PROCESS_H
/*
 *  In this function we do:
 *  1. past our object into background with random scale, flip, transformation based on heatmap_guided_position
 *  2. store the segmentation (rle), class_id, img_id, height, width, bbox into our annotation
 */

#include "coco_mask_extractor.h"
#include "heatmap.h"
#include <random>

class instaBoostConfig
{
public:
    instaBoostConfig(std::vector<std::string> action_candidate, std::vector<float> action_prob, std::vector<float> scale, float dx, float dy, std::vector<int> theta, float color_prob, bool heatmap_flag);
    instaBoostConfig();
    ~instaBoostConfig();

    std::vector<std::string> action_candidate_{"normal", "horizontal", "skip"};
    std::vector<float> action_prob_{1, 0, 0};
    std::vector<float> scale_{0.8, 1.2};
    float dx_ = 15;
    float dy_ = 15;
    std::vector<int> theta_{-1, 1};
    float color_prob_ = 0.5;
    bool heatmap_flag_ = false;

};


class Generator
{
// Random generator based on different probabilities.
public:
    Generator(const float* arrPr, const int length, const unsigned int rand_seed=(unsigned)time(NULL));
    void init_seed(const unsigned int rand_seed);
    int generate();
    virtual ~Generator(); 
private:
    float* prArray;
    int length;
    const float TH = 1e-6;
};
 



std::unordered_map<std::string, float> _get_restriction(std::vector<float> bbox, int width, int height);
std::unordered_map<std::string, float> _identity_transform();
std::unordered_map<std::string, float> _random_transform(std::unordered_map<std::string, float> restricts, instaBoostConfig config);
void _get_transform(cv::Mat *src, std::unordered_map<std::string, float> *t, std::unordered_map<std::string, float> restricts, instaBoostConfig config, std::vector<float> bbox);


#endif
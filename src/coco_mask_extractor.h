#ifndef COCO_MASK_EXTRACTOR_H
#define COCO_MASK_EXTRACTOR_H

#include <vector>
#include <map>
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <string>
#include <fstream>

#include "json.h"

using json = nlohmann::json;

struct Mask 
{
    Mask(int category_id, std::vector<float> segmentation, std::vector<float>bbox)
        : category_id(category_id), segmentation(segmentation), bbox(bbox){}
    // here we only care about those mask that the "iscrowd" is equal to 0
    // TODO: implement all masks that not only "iscrowd" is equal to 0.
    int category_id;
    std::vector<float> bbox;
    std::vector<float> segmentation;
};

struct imgInfo
{

    int width;
    int height;
    std::string file_name;

};

class COCO_mask_extractor
{
    public:
        COCO_mask_extractor(char *json_filename);
        ~COCO_mask_extractor();
        void load();
        void parse();
        std::multimap<int, Mask> & getMaskMultimap();
        std::unordered_map<int, imgInfo> & getImgInfo();


    private:
        std::string json_filename_;
        std::vector<char> raw_json_;
        std::unordered_map<int, imgInfo> imgInfos_multimap_;
        // multimap image_id -> masks
        std::multimap<int, Mask> masks_multimap_;

};

#endif
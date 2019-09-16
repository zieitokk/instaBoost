#include "coco_mask_extractor.h"

/*
 *  In this function we try to extract segmentation annotation from coco .json file
 * 
 *  Since Segmentation annotation in COCO dataset has format:
 *      if "iscrowd" is 0:
 *          Segmentation is [[520.2, 243.4, 511.2, 242.4, ...]], where every consecutive two float number represent x and y of its contour coord.
 *      
 *      else if "iscrowd" is 1:
 *          Segmentation is [81233, 223, 4, 3, ...], which is RLE method for showing the mask.
 *          RLE represent how many zeros at every odd positions and ones at every even positions in this array.
 *       
 *  
 *  Return:
 *      bunch of masks that we stored for "iscrowd" is equal to 0.
 */
COCO_mask_extractor::COCO_mask_extractor(char *json_filename)
{
    json_filename_ = json_filename;
};

COCO_mask_extractor::~COCO_mask_extractor(){};

void COCO_mask_extractor::load()
{
    std::ifstream f(json_filename_);
    f.seekg(0, std::ios::end);
    raw_json_.reserve(f.tellg());
    f.seekg(0, std::ios::beg);
    raw_json_.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void COCO_mask_extractor::parse()
{
    char *source = reinterpret_cast<char *>(raw_json_.data());
    auto j = json::parse(source);
    
    // read image category in COCO dataset
    auto images = j.find("images");
    assert(images != j.end());
    for(auto img: *images)
    {
        int id = img.find("id").value().get<int>();
        std::string filename = img.find("file_name").value().get<std::string>();
        int width = img.find("width").value().get<int>();
        int height = img.find("height").value().get<int>();
        imgInfo imginfo;
        imginfo.width = width;
        imginfo.height = height;
        imginfo.file_name = filename;
        imgInfos_multimap_.insert(std::make_pair(id, imginfo));
    }

    auto annotations = j.find("annotations");
    assert(annotations != j.end());
    for(auto an: *annotations)
    {
        int iscrowd = an.find("iscrowd").value().get<int>();
        if(iscrowd)
            continue;
        int image_id = an.find("image_id").value().get<int>();
        int category_id = an.find("category_id").value().get<int>();
        std::vector<float> bbox = an.find("bbox").value().get<std::vector<float>>();

        std::vector<float> segmentation = an.find("segmentation").value().get<std::vector<std::vector<float>>>()[0];
        masks_multimap_.insert(std::make_pair(image_id, Mask(image_id, segmentation, bbox)));
    }
}

std::multimap<int, Mask> &COCO_mask_extractor::getMaskMultimap()
{
    return masks_multimap_;
}


std::unordered_map<int, imgInfo> &COCO_mask_extractor::getImgInfo()
{
    return imgInfos_multimap_;
}

// // unit test
// int main(int argc, char **argv)
// {
//     COCO_mask_extractor coco_mask_extractor(argv[1]);
//     coco_mask_extractor.load();
//     coco_mask_extractor.parse();
//     auto &ann = coco_mask_extractor.getMaskMultimap();
//     constexpr int img_id = 289343;
//     auto range = ann.equal_range(img_id);
//     std::cout << "Found segmentation for " << img_id << std::endl;
//     for(auto it = range.first; it != range.second; ++it)
//     {
//         Mask an = it ->second;
//         std::cout << "Found segmentation for " << it->first << ": " << an.segmentation[0] << ", " << an.segmentation[1] << std::endl;
//     }

//     return 0;
// }

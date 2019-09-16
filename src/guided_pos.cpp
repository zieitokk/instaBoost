#include "guided_pos.h"

cv::Point _get_paste_pos(cv::Mat heatmap, float ratio)
{
    std::vector<std::vector<int>> poses;
    for(int i = 0; i < heatmap.rows; ++i)
    {
        for(int j = 0; j < heatmap.cols; ++j)
        {
            if(heatmap.at<float>(i, j) > 200)
            {
                std::vector<int> tmp{i, j};
                poses.push_back(tmp);
            }
        }
    }

    float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    if( r < ratio || poses.size() == 0)
    {
        for(int i = 0; i < heatmap.rows; ++i)
        {
            for(int j = 0; j < heatmap.cols; ++j)
            {
                if(heatmap.at<float>(i, j) > 150)
                {
                    std::vector<int> tmp{i, j};
                    poses.push_back(tmp);
                }
            }
        }
    }

    if(poses.size() == 0)
    {
        cv::Point fail;
        fail.x = -1;
        fail.y = -1;
        return fail;
    }

    int pos_size = poses.size();
    int rand_pos = (rand() % (pos_size));
    
    cv::Point pos;
    pos.x = poses[rand_pos][1];
    pos.y = poses[rand_pos][0];
    return pos;
}


// // unit test
// int main(int argc, char **argv)
// {
//     // initialize the coco mask extractor
//     COCO_mask_extractor coco_mask_extractor(argv[1]);
//     coco_mask_extractor.load();
//     coco_mask_extractor.parse();
//     auto &mask_ann = coco_mask_extractor.getMaskMultimap();
//     auto &img_info_ann = coco_mask_extractor.getImgInfo();

//     // now we want to check the image with ID 488673
//     constexpr int img_id = 16502;
//     auto range = mask_ann.equal_range(img_id);
//     imgInfo imginfo = img_info_ann[img_id];
//     // extract its file_name
//     std::string file_name = imginfo.file_name;

//     // std::cout << file_name << std::endl;
//     // // now we get the image from folder
//     // std::vector<std::string> file_name_stack = split(file_name, "_");

//     std::string database = "/media/xiangtao/data/coco2017/val2017/";

//     std::cout << database + file_name << std::endl;
//     cv::Mat img = cv::imread(database + file_name);

//     std::cout << "Found segmentation for " << img_id << std::endl;
//     for(auto it = range.first; it != range.second; ++it)
//     {
//         // for every object we get one heatmap for it.
//         Mask an = it ->second;
//         std::cout << "first check" << std::endl;
//         cv::Mat background = _get_mask_center_background(an, imginfo, img);
//         std::cout << "background" << std::endl;
//         cv::Mat mask = _get_coco_masks(an, imginfo);
//         std::cout << "third check" << std::endl;
//         cv::Mat trimap = _get_trimap(mask, 5);

//         // cv::imshow("trimap", trimap);
//         // cv::waitKey(1000);

//         cv::Point center;
//         center.x = an.bbox[0] + an.bbox[2]/2;
//         center.y = an.bbox[1] + an.bbox[3]/2;
        
//         std::cout << "fifth check" << std::endl;
//         cv::Mat heatmap = _get_heatmap(img, background, trimap, center, 5);
        
//         // normalizedHeatmap is a desHeight x desWidth x 1 float cv::Mat
//         cv::Mat normalizedHeatmap = _normalize(heatmap);
//         cv::Mat normalizedColorHeatmap, tmp;

//         // here we convert it into 8UC cv::Mat
//         normalizedHeatmap.convertTo(tmp, CV_8UC1);
//         cv::applyColorMap(tmp, normalizedColorHeatmap, cv::COLORMAP_JET);
//         cv::resize(normalizedColorHeatmap, normalizedColorHeatmap, cv::Size(img.cols, img.rows), 0.0, 0.0, cv::INTER_LINEAR);
//         cv::imshow("normalized heatmap", normalizedColorHeatmap);
//         cv::waitKey(5000);

//         cv::Point pos;
//         pos = _get_paste_pos(normalizedColorHeatmap, 0.4);
//         std::cout << "paste pos x is: " << pos.x << " y is: " << pos.y << std::endl;
//     }

//     return 0;
// }
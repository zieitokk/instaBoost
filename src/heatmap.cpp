#include "heatmap.h"
#include <math.h>


cv::Mat _get_coco_masks(Mask mask_anno, imgInfo imginfo)
{
    // Return a binary mask (0 or 255), where 255 is the instance that we need.

    std::vector<float> points_vec = mask_anno.segmentation;
    int num_of_polyvertex = mask_anno.segmentation.size()/2;
    cv::Point vertexPoints[num_of_polyvertex];
    for(int i = 0; i < num_of_polyvertex; ++i)
    {
        vertexPoints[i] = cv::Point(points_vec[2*i], points_vec[2*i+1]);
    }

    const cv::Point * pt[1] = {vertexPoints};
    int npt[] = {num_of_polyvertex};

    cv::Mat tmp_copy = cv::Mat::zeros(imginfo.height, imginfo.width, CV_8U);
    cv::fillPoly(tmp_copy, pt, npt, 1, cv::Scalar(255));

    return tmp_copy;
}


cv::Mat _get_trimap(cv::Mat mask, int kernel_size)
{
    cv::Mat trimap = cv::Mat::zeros(mask.rows, mask.cols, CV_32F);
    std::vector<cv::Mat> mask_list;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
    cv::Mat mask_f64;
    mask.convertTo(mask_f64, CV_64F);
    // add original mask first
    mask_list.push_back(mask_f64);
    for(int i = 1; i < 4; ++i)
    {
        cv::Mat dilated_mask;
        cv::dilate(mask_f64, dilated_mask, kernel, cv::Point(-1, -1), i*2);
        // std::cout << dilated_mask << std::endl;
        // cv::imshow("dilated mask", dilated_mask);
        // cv::waitKey(5000);

        mask_list.push_back(dilated_mask);
    }

    for(int i = 3; i > -1; --i)
    {
        for(int row = 0; row < trimap.rows; ++row)
        {
            for(int col = 0; col < trimap.cols; ++col)
            {
                if(mask_list[i].at<double>(row, col) > 200)
                    trimap.at<float>(row, col) = (float)(4 - i);
            }
        }
    }
    // std::cout << trimap << std::endl;
    return trimap;
}


std::deque<std::vector<std::vector<int>>> _get_trimap_coord(cv::Mat trimap)
{
    std::deque<std::vector<std::vector<int>>> trimap_coord;
    for(int i = 1; i < 4; ++i)
    {   
        std::vector<std::vector<int>> tmp_coord;
        int count = 0;
        for(int row = 0; row < trimap.rows; ++row)
        {
            for(int col = 0; col < trimap.cols; ++col)
            {
                if(trimap.at<float>(row, col) == (float)i)
                {   
                    count += 1;
                    std::vector<int> coord{col, row};
                    tmp_coord.push_back(coord);
                }
            }
        }
        trimap_coord.push_back(tmp_coord);
        std::cout << count << std::endl;
    }

    // std::cout << trimap << std::endl;
    return trimap_coord;
}

cv::Mat _get_mask_center_background(Mask mask_anno, imgInfo imginfo, cv::Mat img)
{
    // std::cout << "1" << std::endl;
    cv::Mat img_mask = _get_coco_masks(mask_anno, imginfo);
    // std::cout << img_mask << std::endl;
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    // std::cout << kernel << std::endl;
    cv::dilate(img_mask, img_mask, kernel, cv::Point(-1, -1), 2);
    // std::cout << "4" << std::endl;
    cv::Mat convertedImgMask;
    cv::Mat inpaintedImg;
    img_mask.convertTo(convertedImgMask, CV_8U);

    cv::inpaint(img, convertedImgMask, inpaintedImg, 5, cv::INPAINT_NS);

    return inpaintedImg;
}

std::deque<std::vector<std::vector<int>>> _get_rings(std::deque<std::vector<std::vector<int>>> trimap_coord, cv::Mat img)
{
    // get rings after different dilation process

    int width = img.cols;
    int height = img.rows;
    // printf("img width is: %d and height is: %d", width, height);

    std::deque<std::vector<std::vector<int>>> rings;

    for(int i = 0; i < 3; ++i)
    {
        std::vector<std::vector<int>> coord = trimap_coord.at(i);
        // std::cout << coord.size() << std::endl;
        std::vector<std::vector<int>> tmp;
        bool posflag = true;

        for(int row = 0; row < coord.size(); ++row)
        {
            int x = coord[row][0];
            int y = coord[row][1];
            // printf("x is: %d, y is: %d", x, y);
            if(x >= 0 && y >=0 && x < width && y < height)
            {
                std::vector<int> channels;
                channels.push_back(img.at<cv::Vec3b>(y, x)[0]);
                channels.push_back(img.at<cv::Vec3b>(y, x)[1]);
                channels.push_back(img.at<cv::Vec3b>(y, x)[2]);
                tmp.push_back(channels);

                // printf("channels: %d, %d, %d", channels[0], channels[1], channels[2]);
            }else
            {
                posflag = false;
            }
        }
        if(!posflag)
        {
            std::deque<std::vector<std::vector<int>>> emply_vec;
            return emply_vec;
        }
        rings.push_back(tmp);
    }
    return rings;
}

std::deque<std::vector<std::vector<int>>> _translate_trimap(std::deque<std::vector<std::vector<int>>> trimap_coord, cv::Point oripos, cv::Point augpos)
{
    std::deque<std::vector<std::vector<int>>> augtrimap_coord;
    std::deque<std::vector<std::vector<int>>> augtrimap_coord_out;
    for(int i = 0; i < 3; ++i)
    {
        augtrimap_coord.push_back(trimap_coord.at(i));
    }
    
    for(int i = 0; i < 3; ++i)
    {
        std::vector<std::vector<int>> tmp_aug_coord = augtrimap_coord.at(i);
        std::vector<std::vector<int>> tmp_tri_coord = trimap_coord.at(i);
        for(int row = 0; row < tmp_aug_coord.size(); ++row)
        {
            tmp_aug_coord[row][0] = tmp_tri_coord[row][0] - oripos.x + augpos.x;
            tmp_aug_coord[row][1] = tmp_tri_coord[row][1] - oripos.y + augpos.y;
        }
        augtrimap_coord_out.push_back(tmp_aug_coord);
    }
    return augtrimap_coord_out;
}

float _get_euclidean_distance(std::vector<std::vector<int>> oriRing, std::vector<std::vector<int>> augRing)
{
    std::vector<std::vector<float>> sub_sqr_Ring(oriRing.size());
    for(int i = 0; i < oriRing.size(); ++i)
    {
        sub_sqr_Ring[i].resize(oriRing[0].size());
    }

    for(int i = 0; i < oriRing.size(); ++i)
    {
        for(int j = 0; j < oriRing[0].size(); ++j)
        {
            // printf("ori value is: %f", (float)oriRing[i][j]);
            // printf("aug value is: %f", (float)augRing[i][j]);
            sub_sqr_Ring[i][j] = pow(((float)oriRing[i][j] - (float)augRing[i][j]), 2);
            // printf("ring value is: %f", sub_sqr_Ring[i][j]);
        }
    }

    std::vector<float> sum_and_sqrt_Ring;
    for(int i = 0; i < oriRing.size(); ++i)
    {
        float tmp;
        tmp = sub_sqr_Ring[i][0] + sub_sqr_Ring[i][1] + sub_sqr_Ring[i][2];
        sum_and_sqrt_Ring.push_back(sqrt(tmp));
    }

    float euclidean_distance = 0;
    for(int i = 0; i < oriRing.size(); ++i)
    {
        euclidean_distance += sum_and_sqrt_Ring[i];
    }

    return euclidean_distance;
}

float _get_heat_point(std::deque<std::vector<std::vector<int>>> ori_trimap_coord, std::deque<std::vector<std::vector<int>>> oriRings, cv::Mat background, cv::Point oripos, cv::Point augpos, std::vector<float> config={0.25, 0.35, 0.4})
{
    std::deque<std::vector<std::vector<int>>> aug_trimap_coord;
    aug_trimap_coord = _translate_trimap(ori_trimap_coord, oripos, augpos);
    std::deque<std::vector<std::vector<int>>> aug_rings;
    aug_rings = _get_rings(aug_trimap_coord, background);

    // std::cout << "aug rings size is: " << aug_rings[0].size() << " " << aug_rings[1].size() << " " << aug_rings[2].size() << std::endl;
    if(aug_rings.empty())
        return -1;
    float heatpoint = 0;
    for(int i = 0; i < 3; ++i)
    {
        float ed = _get_euclidean_distance(oriRings.at(i), aug_rings.at(i));
        heatpoint += config[i] * ed;
        // printf("the heat point for rings %d, is %d\n", i, heatpoint);
    }

    return heatpoint;
}


cv::Mat _get_heatmap(cv::Mat img, cv::Mat background, cv::Mat trimap, cv::Point center, int shrink = 10)
{
    /*  
     * Here we will generate our heatmap for our single instance based on its location and the background
     * 
     * Input:
     *      img(cv::Mat): original rgb image
     *      background(cv::Mat): background that our target instance has been cleared from
     *      trimap(cv::Mat): Rings map with 0, 1, 2, 3, 0 from the inner to the outter
     *      center(cv::Point) center location of our target instance
     *      shrink(int): in order to accelerate calculation, we first shrink the map size and then interpolate it back to its original size
     *
     * Output:
     *      heatmap(cv::Mat): probability heatmap for out target instance
     */

    int oriWidth = img.cols;
    int oriHeight = img.rows;
    int desWidth = oriWidth/shrink;
    int desHeight = oriHeight/shrink;

    cv::Mat resized_img, resize_background, resized_trimap;

    cv::resize(img, resized_img, cv::Size(desWidth, desHeight), (0, 0), (0, 0), CV_INTER_LINEAR);
    // cv::imshow("resized img", resized_img);
    // cv::waitKey(5000);

    cv::resize(background, resize_background, cv::Size(desWidth, desHeight), (0, 0), (0, 0), CV_INTER_LINEAR);
    // cv::imshow("resized background", resize_background);
    // cv::waitKey(5000);

    cv::resize(trimap, resized_trimap, cv::Size(desWidth, desHeight), (0, 0), (0, 0), CV_INTER_LINEAR);
    // std::cout << resized_trimap << std::endl;
    // cv::imshow("resized trimap", resized_trimap);
    // cv::waitKey(5000);


    cv::Point oripos;
    oripos.x = center.x / shrink;
    oripos.y = center.y / shrink;

    cv::Mat heatmap = cv::Mat::zeros(desHeight, desWidth, CV_32F);

    std::deque<std::vector<std::vector<int>>> resizedTrimapCoord;
    resizedTrimapCoord = _get_trimap_coord(resized_trimap);
    
    std::deque<std::vector<std::vector<int>>> resizedRingsCoord;
    resizedRingsCoord = _get_rings(resizedTrimapCoord, resized_img);

    // std::cout << resizedTrimapCoord.size() << std::endl;
    // std::cout << resizedRingsCoord.size() << std::endl;
    
    std::vector<std::vector<float>> res;
    for(int i = 0; i < desHeight; ++i)
    {
        for(int j = 0; j < desWidth; ++j)
        {
            float heatPoint;
            std::vector<float> tmp;
            heatPoint = _get_heat_point(resizedTrimapCoord, resizedRingsCoord, resize_background, oripos, cv::Point(j, i));
            tmp.push_back((float)i);
            tmp.push_back((float)j);
            tmp.push_back(heatPoint);
            res.push_back(tmp);
        }
    }

    for(int i = 0; i < res.size(); ++i)
    {
        heatmap.at<float>(res[i][0], res[i][1]) = res[i][2]; 
        // printf("%f ", heatmap.at<float>(res[i][0], res[i][1]));
    }

    return heatmap;
}

cv::Mat _normalize(cv::Mat heatmap)
{
    float max = 0.0;
    for(int i = 0; i < heatmap.rows; ++i)
    {
        for(int j = 0; j < heatmap.cols; ++j)
        {
            if(heatmap.at<float>(i, j) > max)
                max = heatmap.at<float>(i, j);
        }
    }

    std::cout << " max value is: " << max << std::endl;
    // std::cout << heatmap << std::endl;

    cv::Mat cuttedHeatmap = cv::Mat::zeros(heatmap.rows, heatmap.cols, CV_32F);
    for(int i = 0; i < heatmap.rows; ++i)
    {
        for(int j = 0; j < heatmap.cols; ++j)
        {
            if(heatmap.at<float>(i, j) < 0)
            {
                heatmap.at<float>(i, j) = max;
                cuttedHeatmap.at<float>(i, j) = pow(((float)1.0 - heatmap.at<float>(i, j) / max), 3);
            }else
            {
                cuttedHeatmap.at<float>(i, j) = pow(((float)1.0 - heatmap.at<float>(i, j) / max), 3);
            }
        }
    }

    // std::cout << cuttedHeatmap << std::endl;
    max = 0.0;
    for(int i = 0; i < cuttedHeatmap.rows; ++i)
    {
        for(int j = 0; j < cuttedHeatmap.cols; ++j)
        {
            if(cuttedHeatmap.at<float>(i, j) > max)
                max = cuttedHeatmap.at<float>(i, j);
        }
    }
    std::cout << " max value is: " << max << std::endl;

    cv::Mat normalizedHeatmap = cv::Mat::zeros(cuttedHeatmap.rows, cuttedHeatmap.cols, CV_32F);
    for(int i = 0; i < cuttedHeatmap.rows; ++i)
    {
        for(int j = 0; j < cuttedHeatmap.cols; ++j)
        {
            normalizedHeatmap.at<float>(i, j) = cuttedHeatmap.at<float>(i, j) * 255 / max;
        }
    }

    // std::cout << normalizedHeatmap << std::endl;
    return normalizedHeatmap;
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
//     constexpr int img_id = 885;
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
//         cv::Mat heatmap = _get_heatmap(img, background, trimap, center);
        
//         // normalizedHeatmap is a desHeight x desWidth x 1 float cv::Mat
//         cv::Mat normalizedHeatmap = _normalize(heatmap);
//         cv::Mat normalizedColorHeatmap, tmp;

//         // here we convert it into 8UC cv::Mat
//         normalizedHeatmap.convertTo(tmp, CV_8UC1);
//         cv::applyColorMap(tmp, normalizedColorHeatmap, cv::COLORMAP_JET);
//         cv::resize(normalizedColorHeatmap, normalizedColorHeatmap, cv::Size(img.cols, img.rows), 0.0, 0.0, cv::INTER_LINEAR);
//         cv::imshow("normalized heatmap", normalizedColorHeatmap);
//         cv::waitKey(5000);
//     }

//     return 0;
// }
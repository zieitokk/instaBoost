#include "affine_transform.h"
#include <time.h>

void _transform(cv::Mat *target, Mask *an, cv::Mat *new_img, cv::Mat background, std::unordered_map<std::string, float> t, cv::Point guided_pos)
{
    /*
     * First we need to transform the annotation based on the t:
     *      t["flip"] = 10 ==> "horizontal flip" and the segmentation should be reversed
     *      t["flip"] = 11 ==> "vertical flip" and the segmentation should be reversed
     * 
     * Second we move the target center at guided_pos.
     * Third we shift the target with t["tx"] for x-coord and t["ty"] for y-coord
     * Then we scale and rotate the target to get the final one.
     * And the last, we need to check the target's four conner coords and segmentation points are within the img.width x img.height
     * 
     */
    std::vector<float> guided_bbox = an -> bbox;
    std::vector<float> transformed_bbox_cxcywh{0,0,0,0};
    cv::Rect roi = cv::Rect(floor(guided_bbox[0]+0.5), floor(guided_bbox[1]+0.5), floor(guided_bbox[2]+0.5), floor(guided_bbox[3]+0.5));

    // rotate, scale and shift the target
    cv::Mat M = cv::Mat::zeros(2, 3, CV_32F);
    M.at<float>(0, 0) = t["s"] * cos(t["theta"]);
    M.at<float>(0, 1) = t["s"] * sin(t["theta"]);
    M.at<float>(0, 2) = guided_pos.x + t["tx"] - guided_bbox[0] - guided_bbox[2]/2;
    M.at<float>(1, 0) = -1. * t["s"] * sin(t["theta"]);
    M.at<float>(1, 1) = t["s"] * cos(t["theta"]);
    M.at<float>(1, 2) = guided_pos.y + t["ty"] - guided_bbox[1] - guided_bbox[3]/2;

    cv::warpAffine(*target, (*target), M, cv::Size(target -> cols, target -> rows));
    // cv::imshow("rotated_target", (*target));
    // cv::waitKey(5000);

    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::Mat gray_img;
    cv::cvtColor((*target), gray_img, CV_BGR2GRAY);
    cv::findContours(gray_img, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    // create new annotations for current image
    std::vector<float> new_bbox;
    std::vector<float> new_segmentation;
    float xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    for(int i = 0; i < contours[0].size(); ++i)
    {
        new_segmentation.push_back(contours[0][i].x);
        new_segmentation.push_back(contours[0][i].y);
        if(xmin > contours[0][i].x)
            xmin = contours[0][i].x;
        if(xmax < contours[0][i].x)
            xmax = contours[0][i].x;
        if(ymin > contours[0][i].y)
            ymin = contours[0][i].y;
        if(ymax < contours[0][i].y)
            ymax = contours[0][i].y;
    }

    new_bbox = std::vector<float>{xmin, ymin, xmax - xmin, ymax - ymin};
    an -> bbox = new_bbox;
    an -> segmentation = new_segmentation;

    // std::cout << contours.size() << " " << contours[0].size() << std::endl;

    cv::Mat mask_image( target -> size(), CV_8U, cv::Scalar(0));
    cv::drawContours(mask_image, contours, -1, cv::Scalar(255), -1);
    // cv::imshow("mask img", mask_image);
    // cv::waitKey(5000);
    *new_img = background;
    target -> copyTo((*new_img), mask_image);
    // cv::imshow("final combined img", (*new_img));
    // cv::waitKey(5000);
}


// unit test
int main(int argc, char **argv)
{
    // initialize the coco mask extractor
    COCO_mask_extractor coco_mask_extractor(argv[1]);
    coco_mask_extractor.load();
    coco_mask_extractor.parse();
    auto &mask_ann = coco_mask_extractor.getMaskMultimap();
    auto &img_info_ann = coco_mask_extractor.getImgInfo();

    // now we want to check the image with ID 16502
    constexpr int img_id = 885;
    auto range = mask_ann.equal_range(img_id);
    imgInfo imginfo = img_info_ann[img_id];
    // extract its file_name
    std::string file_name = imginfo.file_name;

    // std::cout << file_name << std::endl;
    // // now we get the image from folder
    // std::vector<std::string> file_name_stack = split(file_name, "_");

    std::string database = "/media/xiangtao/data/coco2017/val2017/";

    std::cout << database + file_name << std::endl;
    cv::Mat img = cv::imread(database + file_name);

    std::cout << "Found segmentation for " << img_id << std::endl;

    for(auto it = range.first; it != range.second; ++it)
    {
        // for every object we get one heatmap for it.
        clock_t start = clock();
        Mask an = it ->second;
        cv::Mat background = _get_mask_center_background(an, imginfo, img);
        // std::cout << "background" << std::endl;
        cv::Mat mask = _get_coco_masks(an, imginfo);
        // std::cout << "third check" << std::endl;
        cv::Mat trimap = _get_trimap(mask, 5);

        // cv::imshow("trimap", trimap);
        // cv::waitKey(1000);

        cv::Point center;
        center.x = an.bbox[0] + an.bbox[2]/2;
        center.y = an.bbox[1] + an.bbox[3]/2;
        
        // std::cout << "fifth check" << std::endl;
        cv::Mat heatmap = _get_heatmap(img, background, trimap, center, 10);
        
        // normalizedHeatmap is a desHeight x desWidth x 1 float cv::Mat
        cv::Mat normalizedHeatmap = _normalize(heatmap);
        cv::Mat normalizedColorHeatmap, tmp;

        // here we convert it into 8UC cv::Mat
        normalizedHeatmap.convertTo(tmp, CV_8UC1);
        cv::applyColorMap(tmp, normalizedColorHeatmap, cv::COLORMAP_JET);
        cv::resize(normalizedColorHeatmap, normalizedColorHeatmap, cv::Size(img.cols, img.rows), 0.0, 0.0, cv::INTER_LINEAR);
        cv::imshow("normalized heatmap", normalizedColorHeatmap);
        cv::waitKey(5000);

        cv::Point pos;
        pos = _get_paste_pos(normalizedColorHeatmap, 0.8);
        std::cout << "paste pos x is: " << pos.x << " y is: " << pos.y << std::endl;


        // and for each segmentation, we extract the target and randomly choose whether we transform it or not
        cv::Mat target;
        img.copyTo(target, mask);
        // cv::imshow("target", target);
        // cv::waitKey(5000);
        
        instaBoostConfig config;
        config.action_candidate_ = std::vector<std::string>{"normal", "vertical", "skip"};
        config.action_prob_ = std::vector<float>{0.2, 0.7, 0.1};
        std::unordered_map<std::string, float> restricts, t;
        restricts = _get_restriction(an.bbox, imginfo.width, imginfo.height);
        _get_transform(&target, &t, restricts, config, an.bbox);

        std::cout << t["tx"] << "/" << t["ty"] << "/" << t["theta"] << std::endl;

        cv::Mat new_img;
        _transform(&target, &an, &new_img, background, t, pos);

        cv::imshow("transformed target", new_img);
        cv::waitKey(5000);
        clock_t end = clock();
        std::cout << "total time consumed is: " << (double)(end - start)/CLOCKS_PER_SEC << " seconds" << std::endl;
    }

    return 0;
}
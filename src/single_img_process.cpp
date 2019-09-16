#include "single_img_process.h"


instaBoostConfig::instaBoostConfig(std::vector<std::string> action_candidate, std::vector<float> action_prob, std::vector<float> scale, float dx, float dy, std::vector<int> theta, float color_prob, bool heatmap_flag)
{
    action_candidate_ = action_candidate;
    action_prob_ = action_prob;
    scale_ = scale;
    dx_ = dx;
    dy_ = dy;
    theta_ = theta;
    color_prob_ = color_prob;
    heatmap_flag_ = heatmap_flag;
};
instaBoostConfig::instaBoostConfig(){};
instaBoostConfig::~instaBoostConfig(){};

Generator::Generator(const float* arrPr, const int length, const unsigned int rand_seed)
{
    float sum=0.0;
    for(int i=0;i<length;i++)
    {
        sum+=arrPr[i];
    }
    if(fabs(sum-1)>TH) throw "The sum of probabilities does not match 1.";
     
    this->length = length;
    prArray = new float[length];
    for(int i=0;i<length;i++)
    {
        if(i==0) prArray[0]=arrPr[0];
        else prArray[i]=prArray[i-1]+arrPr[i];
    }
    init_seed(rand_seed);
}
 
Generator::~Generator()
{
    delete[] prArray;
}
 
void Generator::init_seed(const unsigned int rand_seed)
{
    srand(rand_seed);
}
 
int Generator::generate()
{
    float randnum = (float)rand()/RAND_MAX;
    int i;
    for(i=0;i<length-1&&randnum-prArray[i]>TH;i++);
    return i;
}

std::unordered_map<std::string, float> _get_restriction(std::vector<float> bbox, int width, int height)
{
    float xmin = bbox[0];
    float ymin = bbox[1];
    float w = bbox[2];
    float h = bbox[3];
    float xmax = xmin + w;
    float ymax = ymin + h;

    std::unordered_map<std::string, float> restricts;
    restricts.insert({"bbox_w", w});
    restricts.insert({"bbox_h", h});

    if(xmin < 10)
    {
        restricts.insert({"restrict_left", 1});
        restricts.insert({"noflip", 1});
    }
    if(xmax > (width - 10))
    {
        restricts.insert({"restrict_right", 1});
        if(restricts.find("noflip") != restricts.end())
        {   
            restricts["noflip"] = 1;
        }
        else
        {
            restricts.insert({"noflip", 1});           
        }  
    }
    if(ymin < 10)
    {
        restricts.insert({"restrict_up", 1});
        if(restricts.find("noflip") != restricts.end())
        {   
            restricts["noflip"] = 1;
        }
        else
        {
            restricts.insert({"noflip", 1});           
        } 
    }
    if(ymax > (height - 10))
    {
        restricts.insert({"restrict_down", 1});
        if(restricts.find("noflip") != restricts.end())
        {   
            restricts["noflip"] = 1;
        }
        else
        {
            restricts.insert({"noflip", 1});           
        }
    }

    return restricts;
}

std::unordered_map<std::string, float> _identity_transform()
{
    std::unordered_map<std::string, float> t;
    t.insert({"s", 1});
    t.insert({"tx", 0});
    t.insert({"ty", 0});
    t.insert({"theta", 0});
    return t;
}

std::unordered_map<std::string, float> _random_transform(std::unordered_map<std::string, float> restricts, instaBoostConfig config)
{
    std::unordered_map<std::string, float> t;
    std::random_device rd;
    std::default_random_engine e{rd()};
    std::uniform_real_distribution<float> us(config.scale_[0], config.scale_[1]);
    float rs = us(e);
    t.insert({"s", rs});

    float max_x = floor(restricts["bbox_w"] / config.dx_);
    std::uniform_real_distribution<float> utx(-max_x, max_x);
    float rtx = utx(e);
    t.insert({"tx", rtx});

    float max_y = floor(restricts["bbox_h"] / config.dy_);
    std::uniform_real_distribution<float> uty(-max_y, max_y);
    float rty = uty(e);
    t.insert({"ty", rty});

    std::uniform_int_distribution<> utheta(config.theta_[0], config.theta_[1]);
    int dtheta = utheta(e);
    float pi = acos(-1);
    float rtheta = dtheta * pi / 180;
    t.insert({"theta", rtheta});

    if(restricts.find("restrict_left") != restricts.end() && restricts["restrict_left"] == 1)
    {
        t["s"] = t["s"] > 1?t["s"]:1;
        t["tx"] = t["tx"] < 0?t["tx"]:0;
        t["theta"] = 0;
    }
    if(restricts.find("restrict_right") != restricts.end() && restricts["restrict_right"] == 1)
    {
        t["s"] = t["s"] > 1?t["s"]:1;
        t["tx"] = t["tx"] > 0?t["tx"]:0;
        t["theta"] = 0;    
    }
    if(restricts.find("restrict_up") != restricts.end() && restricts["restrict_up"] == 1)
    {
        t["s"] = t["s"] > 1?t["s"]:1;
        t["ty"] = t["ty"] < 0?t["tx"]:0;
        t["theta"] = 0;       
    }
    if(restricts.find("restrict_down") != restricts.end() && restricts["restrict_down"] == 1)
    {
        t["s"] = t["s"] > 1?t["s"]:1;
        t["ty"] = t["ty"] > 0?t["tx"]:0;
        t["theta"] = 0;
    }
    return t;
}

void _get_transform(cv::Mat *src, std::unordered_map<std::string, float> *t, std::unordered_map<std::string, float> restricts, instaBoostConfig config, std::vector<float>bbox)
{
    std::vector<std::string> action_candidate = config.action_candidate_;
    std::vector<float> action_prob = config.action_prob_;

    if(restricts.find("noflip") != restricts.end() && restricts["noflip"] == 1)
    {
        std::vector<std::string>::iterator it = action_candidate.begin();
        std::vector<float>::iterator itt = action_prob.begin();
        for(; it != action_candidate.end();)
        {
            if(*it == "horizontal" || *it == "vertical")
            {
                it = action_candidate.erase(it);
                itt = action_prob.erase(itt);
            }else
            {
                it++;
                itt++;
            }
        }
    }

    // since we cut the prob, and the sum of prob now is not equal to 1.
    // so we need to distribute those prob with ratio
    float sum_prob = 0;
    for(int i = 0; i < action_prob.size(); ++i)
    {
        sum_prob += action_prob[i];
    }
    for(int i = 0; i < action_prob.size(); ++i)
    {
        action_prob[i] = action_prob[i] / sum_prob;
    }

    Generator g(action_prob.data(), action_candidate.size());
    std::string action_what = action_candidate[g.generate()];
    if(action_what == "skip")
        *t = _identity_transform();
    else if (action_what == "horizontal")
    {
        auto roi = cv::Rect(floor(bbox[0]+0.5), floor(bbox[1]+0.5), floor(bbox[2]+0.5), floor(bbox[3]+0.5));
        cv::flip((*src)(roi), (*src)(roi), 0);
        *t = _random_transform(restricts, config);
        // here since our map is stored with key: string and value: float.
        // So we decide to use 10 as "horizontal", 11 as "vertical"
        t -> insert({"flip", 10});
    }else if (action_what == "vertical")
    {
        auto roi = cv::Rect(floor(bbox[0]+0.5), floor(bbox[1]+0.5), floor(bbox[2]+0.5), floor(bbox[3]+0.5));
        cv::flip((*src)(roi), (*src)(roi), 1);
        *t = _random_transform(restricts, config);
        t -> insert({"flip", 11});
    }else if (action_what == "normal")
    {
        *t = _random_transform(restricts, config);
    }else
    {
        throw "Unknown action!";
    }
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

//     // now we want to check the image with ID 16502
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
//         // and for each segmentation, we extract the target and randomly choose whether we transform it or not
//         cv::Mat mask = _get_coco_masks(an, imginfo);
//         cv::Mat target;
//         img.copyTo(target, mask);
//         cv::imshow("target", target);
//         cv::waitKey(5000);
        
//         instaBoostConfig config;
//         config.action_prob_ = std::vector<float>{0.2, 0.7, 0.1};
//         std::unordered_map<std::string, float> restricts, t;
//         restricts = _get_restriction(an.bbox, imginfo.width, imginfo.height);
//         _get_transform(&target, &t, restricts, config, an.bbox);

//         cv::imshow("transformed target", target);
//         cv::waitKey(5000);
//         std::cout << t["tx"] << "/" << t["ty"] << "/" << t["theta"] << std::endl;
//     }

//     return 0;
// }
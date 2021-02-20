#include <iostream>
#include <filesystem>
#include <mutex>

#include <opencv2/opencv.hpp>

static std::mutex busy;

cv::Mat ContrastBrightnessGrayscale(const cv::Mat& image, double a = 1.0, double b = 0.5)
{
    cv::Mat new_image = cv::Mat::zeros(image.size(), image.type());
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            double color_normalized = static_cast<double>(image.at<uchar>(y, x)) / 255.;  // [0, 255] -> [0, 1]
            double new_color_normalized = (color_normalized - b) * a + b;
            int new_color = static_cast<int>(new_color_normalized * 255);
            new_color = std::min(new_color, 255);
            new_color = std::max(new_color, 0);
            new_image.at<uchar>(y, x) = static_cast<uchar>(new_color);
        }
    }
    return std::move(new_image);
}

cv::Mat Gray2BW(const cv::Mat& image, int threshold)
{
    cv::Mat new_image;
    image.copyTo(new_image);
    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            auto& c = new_image.at<uchar>(y, x);
            if (c < threshold)
                c = 0;
            else
                c = 255;
        }
    }
    return std::move(new_image);
}

cv::Rect BoundingBox(const cv::Mat& image, const uchar threshold_min = 0, const uchar threshold_max = 255, const int border = 0)
{
    int min_x = image.cols;
    int max_x = 0;
    int min_y = image.rows;
    int max_y = 0;

    for (int y = 0; y < image.rows; y++)
    {
        for (int x = 0; x < image.cols; x++)
        {
            auto& p = image.at<uchar>(y, x);
            if (p >= threshold_min && p <= threshold_max)
            {
                if (min_x > x)
                    min_x = x;
                if (max_x < x)
                    max_x = x;
                if (min_y > y)
                    min_y = y;
                if (max_y < y)
                    max_y = y;
            }
        }
    }

    if (min_x > max_x || min_y > max_y)
        return cv::Rect(0, 0, 0, 0);

    if (border != 0)
    {
        min_x = std::max(min_x - border, 0);
        max_x = std::min(max_x + border, image.cols - 1);
        min_y = std::max(min_y - border, 0);
        max_y = std::min(max_y + border, image.rows - 1);
    }

    return cv::Rect(min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
}

cv::Size new_img_size(const cv::Mat& input_image, const int target_image_size)
{
    cv::Size img_size = input_image.size();
    auto& more = img_size.height > img_size.width ? img_size.height : img_size.width;
    auto& less = img_size.height > img_size.width ? img_size.width : img_size.height;
    less = int(double(less) / more * target_image_size);
    more = target_image_size;
    return std::move(img_size);
}

struct Parameters
{
    std::vector<cv::Mat> res;
    cv::Mat lens_rectangle_mask;
    float lens_zoom = 1.f;
    int target_image_size = 10000;
    int target_image_size_min = 10;
    double gaussian_kernel_size_percent = 1.5;
    int gaussian_kernel_size = 1;
    int threshold_black = 0;
    int threshold_white = 240;
    int border_in_source = 10;
    double approximate_resizing_rate;
    int contrast_alpha = 10000;
    int contrast_beta = 60;
    int bilateralFilter_d = 15;
    int bilateralFilter_sigmaColor = 80;
    int bilateralFilter_sigmaSpace = 80;
    int binary_threshold_maxval = 127;
    void Reset()
    {
        Parameters default_params;
        lens_zoom = default_params.lens_zoom;
        target_image_size = default_params.target_image_size;
        target_image_size_min = default_params.target_image_size_min;
        gaussian_kernel_size_percent = default_params.gaussian_kernel_size_percent;
        threshold_black = default_params.threshold_black;
        threshold_white = default_params.threshold_white;
        border_in_source = default_params.border_in_source;
        approximate_resizing_rate = default_params.approximate_resizing_rate;
        contrast_alpha = default_params.contrast_alpha;
        contrast_beta = default_params.contrast_beta;
        bilateralFilter_d = default_params.bilateralFilter_d;
        bilateralFilter_sigmaColor = default_params.bilateralFilter_sigmaColor;
        bilateralFilter_sigmaSpace = default_params.bilateralFilter_sigmaSpace;
        binary_threshold_maxval = default_params.binary_threshold_maxval;
        gaussian_kernel_size = static_cast<int>(target_image_size / (gaussian_kernel_size_percent * 200.)) * 2 + 1;
    }
};

int FindLatestImage(const Parameters& param)
{
    int i = param.res.size();
    do --i; while (i >= 0 && param.res[i].empty());
    return i;
}

bool RemoveLensMargin(Parameters& param, int img_index, bool clear)
{
    if (!param.lens_rectangle_mask.empty())
    {
        if (img_index < 0)
            img_index = FindLatestImage(param);
        if (img_index >= 0)
        {
            cv::bitwise_xor(param.lens_rectangle_mask, param.res[img_index], param.res[img_index]);
            if (clear)
                param.lens_rectangle_mask.release();
            else
                param.lens_rectangle_mask.setTo(cv::Scalar(0));
            return true;
        }
    }
    return false;
}

void process(int stage, void* data);

void process_1(int, void* data)
{
    Parameters& param = *reinterpret_cast<Parameters*>(data);
    cv::bilateralFilter(param.res[0], param.res[1], param.bilateralFilter_d, static_cast<double>(param.bilateralFilter_sigmaColor), static_cast<double>(param.bilateralFilter_sigmaSpace));
}

void process_2(int, void* data)
{
    Parameters& param = *reinterpret_cast<Parameters*>(data);
    cv::copyMakeBorder(param.res[1], param.res[2], param.border_in_source, param.border_in_source, param.border_in_source, param.border_in_source, cv::BORDER_CONSTANT, 255);
    auto bounding_box = BoundingBox(param.res[2], param.threshold_black, param.threshold_white, param.border_in_source);
    if (!bounding_box.empty())
        param.res[2] = param.res[2](bounding_box);
}

void process_3(int, void* data)
{
    Parameters& param = *reinterpret_cast<Parameters*>(data);
    if (param.target_image_size < param.target_image_size_min)
        param.target_image_size = param.target_image_size_min;
    param.approximate_resizing_rate = double(param.target_image_size) / std::max(param.res[0].rows, param.res[0].cols);
    cv::resize(param.res[2], param.res[3], new_img_size(param.res[2], param.target_image_size + static_cast<int>(param.approximate_resizing_rate * param.border_in_source * 2)), 0., 0., cv::INTER_CUBIC);
}

void process_4(int, void* data)
{
    Parameters& param = *reinterpret_cast<Parameters*>(data);
    if (param.gaussian_kernel_size % 2 == 0)
        ++param.gaussian_kernel_size;
    cv::copyMakeBorder(param.res[3], param.res[4], param.gaussian_kernel_size / 2, param.gaussian_kernel_size / 2, param.gaussian_kernel_size / 2, param.gaussian_kernel_size / 2, cv::BORDER_CONSTANT, 255);
    cv::GaussianBlur(param.res[4], param.res[4], { param.gaussian_kernel_size, param.gaussian_kernel_size }, 0.);
}

void process_5(int, void* data)
{
    Parameters& param = *reinterpret_cast<Parameters*>(data);
    param.res[5] = ContrastBrightnessGrayscale(param.res[4], static_cast<double>(param.contrast_alpha) / 100., static_cast<double>(param.contrast_beta) / 100.);
}

void process_6(int, void* data)
{
    Parameters& param = *reinterpret_cast<Parameters*>(data);
    auto bounding_box = BoundingBox(param.res[5], param.threshold_black, param.threshold_white);
    if (bounding_box.empty())
        param.res[5].copyTo(param.res[6]);
    else
        param.res[6] = param.res[5](bounding_box);
}

void process_7(int, void* data)
{
    Parameters& param = *reinterpret_cast<Parameters*>(data);
    param.res[7] = Gray2BW(param.res[6], param.binary_threshold_maxval);
}

void process(int stage, void* data)
{
    Parameters& param = *reinterpret_cast<Parameters*>(data);
    RemoveLensMargin(param, -1, true);
    if (param.res.size() < stage || param.res[stage - 1].empty())
        process(stage - 1, data);
    param.res.resize(stage + 1);

    switch (stage)
    {
    case 1: process_1(0, data); break;
    case 2: process_2(0, data); break;
    case 3: process_3(0, data); break;
    case 4: process_4(0, data); break;
    case 5: process_5(0, data); break;
    case 6: process_6(0, data); break;
    case 7: process_7(0, data); break;
    default: break;
    }

    cv::imshow("Output", param.res[stage]);
}

static void param_reset(int, void* data)
{
    Parameters& param = *reinterpret_cast<Parameters*>(data);
    param.Reset();
}

#define STAGE(i)\
static void stage_##i(int, void* data)\
{\
    std::unique_lock<std::mutex> guard(busy, std::try_to_lock);\
    if (!guard.owns_lock())\
        return;\
    process(i, data);\
}\

STAGE(1)
STAGE(2)
STAGE(3)
STAGE(4)
STAGE(5)
STAGE(6)
STAGE(7)

void CorrectROI(cv::Rect& roi, const cv::Mat& image)
{
    roi.width = std::min(roi.width, image.cols);
    roi.height = std::min(roi.height, image.rows);
    roi.x = std::max(roi.x, 0);
    roi.y = std::max(roi.y, 0);
    roi.x = std::min(roi.x, image.cols - roi.width);
    roi.y = std::min(roi.y, image.rows - roi.height);
}

void CallBackFunc(int event, int x, int y, int flags, void* data)
{
    if (event == cv::EVENT_MOUSEMOVE || event == cv::EVENT_MOUSEWHEEL)
    {
        Parameters& param = *reinterpret_cast<Parameters*>(data);
        int i = FindLatestImage(param);
        if (i >= 0)
        {
            if (event == cv::EVENT_MOUSEWHEEL)
            {
                if (cv::getMouseWheelDelta(flags) > 0)
                    param.lens_zoom *= 2.f;
                else
                    param.lens_zoom *= 0.5f;
                //std::cout << "Zoom: x" << std::to_string(param.lens_zoom) << std::endl;
            }
            cv::Rect roi = cv::getWindowImageRect("Lens");
            roi.width = int(float(roi.width) / param.lens_zoom);
            roi.height = int(float(roi.height) / param.lens_zoom);
            roi.x = x - roi.width / 2;
            roi.y = y - roi.height / 2;
            CorrectROI(roi, param.res[i]);

            if (!RemoveLensMargin(param, i, false))
                param.lens_rectangle_mask = cv::Mat(param.res[i].size(), param.res[i].type(), cv::Scalar(0));
            cv::imshow("Lens", cv::Mat(param.res[i], roi));
            cv::rectangle(param.lens_rectangle_mask, roi, cv::Scalar(255), param.res[i].cols / 200);
            cv::bitwise_xor(param.lens_rectangle_mask, param.res[i], param.res[i]);
            cv::imshow("Output", param.res[i]);
        }
    }
}

void process_file(const std::filesystem::path& input_file, const std::filesystem::path& output_file)
{
    std::cout << input_file << " ... ";
    Parameters param;
    param.res.push_back(cv::imread(input_file.generic_string()));
    if (param.res.empty() || param.res[0].empty() || param.res[0].size().height * param.res[0].size().width == 0)
    {
        std::cout << "ERROR\n";
        return;
    }
    param.Reset();
    cv::cvtColor(param.res[0], param.res[0], cv::COLOR_BGR2GRAY);
    if (param.res[0].cols > param.res[0].rows)
        cv::rotate(param.res[0], param.res[0], cv::ROTATE_90_COUNTERCLOCKWISE);
    cv::namedWindow("Input", cv::WINDOW_KEEPRATIO);
    cv::imshow("Input", param.res[0]);
    cv::namedWindow("Output", cv::WINDOW_KEEPRATIO);
    cv::setMouseCallback("Output", CallBackFunc, reinterpret_cast<void*>(&param));
    cv::namedWindow("Lens", cv::WINDOW_KEEPRATIO);
    process(7, reinterpret_cast<void*>(&param));

    cv::namedWindow("Controls", cv::WINDOW_FREERATIO);
    //cv::resizeWindow("Controls", 200, 1000);
    cv::createTrackbar("Reset", "Controls", nullptr, 1, param_reset, reinterpret_cast<void*>(&param));
    cv::createTrackbar("0:d", "Controls", &param.bilateralFilter_d, 100, stage_1, reinterpret_cast<void*>(&param));
    cv::createTrackbar("0:color", "Controls", &param.bilateralFilter_sigmaColor, 100, stage_1, reinterpret_cast<void*>(&param));
    cv::createTrackbar("0:space", "Controls", &param.bilateralFilter_sigmaSpace, 100, stage_1, reinterpret_cast<void*>(&param));
    cv::createTrackbar("1:black", "Controls", &param.threshold_black, 255, stage_2, reinterpret_cast<void*>(&param));
    cv::createTrackbar("1:white", "Controls", &param.threshold_white, 255, stage_2, reinterpret_cast<void*>(&param));
    cv::createTrackbar("1:brdr", "Controls", &param.border_in_source, 150, stage_2, reinterpret_cast<void*>(&param));
    cv::createTrackbar("2:ImgSize", "Controls", &param.target_image_size, 20000, stage_3, reinterpret_cast<void*>(&param));
    cv::createTrackbar("3:Gauss", "Controls", &param.gaussian_kernel_size, 500, stage_4, reinterpret_cast<void*>(&param));
    cv::createTrackbar("4:Alpha", "Controls", &param.contrast_alpha, 25500, stage_5, reinterpret_cast<void*>(&param));
    cv::createTrackbar("4:Beta", "Controls", &param.contrast_beta, 100, stage_5, reinterpret_cast<void*>(&param));
    cv::createTrackbar("6:To BW", "Controls", &param.binary_threshold_maxval, 256, stage_7, reinterpret_cast<void*>(&param));
    cv::createTrackbar("ProcessAll", "Controls", nullptr, 1, stage_7, reinterpret_cast<void*>(&param));

    cv::waitKey(0);

    if (param.res.size() != 8 || param.res[7].empty())
        process(7, reinterpret_cast<void*>(&param));
    if (param.res.size() != 8 || param.res[7].empty())
    {
        std::cout << "ERROR\n";
        return;
    }
    else
    {
        RemoveLensMargin(param, -1, true);
        cv::imwrite(output_file.generic_string(), param.res[7]);
        std::cout << "OK\n";
        return;
    }
}

int main(int argc, char** argv) try
{
    if (argc != 3)
        throw "Usage:\n\t<exe> <input path> <output path>";

    const std::filesystem::path input_path(argv[1]);
    const std::filesystem::path output_path(argv[2]);
    if (!std::filesystem::is_directory(input_path))
    {
        process_file(input_path, output_path);
        return 0;
    }
    
    std::filesystem::create_directories(output_path);
    for (auto& p : std::filesystem::directory_iterator(input_path))
    {
        process_file(p.path(), output_path / p.path().filename().replace_extension(".png"));
        //break;
    }
    return 0;
}
catch (const std::exception& e)
{
    std::cout << "Error: " << e.what() << "\n";
    return 1;
}
catch (const std::runtime_error& e)
{
    std::cout << "Error: " << e.what() << "\n";
    return 1;
}
catch (const std::string& e)
{
    std::cout << "Error: " << e << "\n";
    return 1;
}
catch (const char* e)
{
    std::cout << "Error: " << e << "\n";
    return 1;
}
catch (...)
{
    std::cout << "Error: UNKNOWN\n";
    return 1;
}

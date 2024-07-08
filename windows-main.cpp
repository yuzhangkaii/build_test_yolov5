#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <onnxruntime_cxx_api.h>
#include <cstdlib>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace Ort;

struct Net_config
{
    float confThreshold; // 置信度阈值
    float nmsThreshold;  // 非极大值抑制阈值
    float objThreshold;  // 物体置信度阈值
    string model_path;
    string classesFile;
};

typedef struct BoxInfo
{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;

class YOLO
{
public:
    YOLO(Net_config config);
    void detect(Mat& frame);
// 添加获取检测框信息的方法
    vector<BoxInfo> getDetectedBoxes() const {
        return this->detected_boxes;
    }
private:
    const float anchors[3][6] = {{10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0}, {116.0, 90.0, 156.0, 198.0, 373.0, 326.0}};
    const float stride[3] = {8.0, 16.0, 32.0};
    int inpWidth;
    int inpHeight;
    vector<string> class_names;
    int num_class;
    float confThreshold;
    float nmsThreshold;
    float objThreshold;

    Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
    vector<float> input_image_;
    void normalize_(Mat img);
    void nms(vector<BoxInfo>& input_boxes);

    Env env = Env(ORT_LOGGING_LEVEL_ERROR, "yolov5-lite");
    Ort::Session *ort_session = nullptr;
    SessionOptions sessionOptions = SessionOptions();
    vector<char*> input_names;
    vector<char*> output_names;
    vector<vector<int64_t>> input_node_dims;
    vector<vector<int64_t>> output_node_dims;
    vector<BoxInfo> detected_boxes; // 存储检测到的边界框信息
};

YOLO::YOLO(Net_config config)
{
    this->confThreshold = config.confThreshold;
    this->nmsThreshold = config.nmsThreshold;
    this->objThreshold = config.objThreshold;

    string model_path = config.model_path;
    std::wstring widestr = std::wstring(model_path.begin(), model_path.end());
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    ort_session = new Session(env, widestr.c_str(), sessionOptions);
    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < numInputNodes; i++)
    {
        input_names.push_back(ort_session->GetInputName(i, allocator));
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
        output_names.push_back(ort_session->GetOutputName(i, allocator));
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }
    this->inpHeight = input_node_dims[0][2];
    this->inpWidth = input_node_dims[0][3];
    string classesFile = config.classesFile;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line))
        this->class_names.push_back(line);
    this->num_class = class_names.size();
}

Mat YOLO::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
    int srch = srcimg.rows, srcw = srcimg.cols;
    *newh = this->inpHeight;
    *neww = this->inpWidth;
    Mat dstimg;
    if (srch != srcw)
    {
        float hw_scale = (float)srch / srcw;
        if (hw_scale > 1)
        {
            *newh = this->inpHeight;
            *neww = int(this->inpWidth / hw_scale);
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *left = int((this->inpWidth - *neww) * 0.5);
            copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, 0);
        }
        else
        {
            *newh = (int)this->inpHeight * hw_scale;
            *neww = this->inpWidth;
            resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
            *top = (int)(this->inpHeight - *newh) * 0.5;
            copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, 0);
        }
    }
    else
    {
        resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
    }
    return dstimg;
}

void YOLO::normalize_(Mat img)
{
    int row = img.rows;
    int col = img.cols;
    this->input_image_.resize(row * col * img.channels());
    for (int c = 0; c < 3; c++)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                float pix = img.ptr<uchar>(i)[j * 3 + 2 - c];
                this->input_image_[c * row * col + i * col + j] = pix / 255.0;
            }
        }
    }
}

void YOLO::nms(vector<BoxInfo>& input_boxes)
{
    sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; });
    vector<float> vArea(input_boxes.size());
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        vArea[i] = (input_boxes.at(i).x2 - input_boxes.at(i).x1 + 1) * (input_boxes.at(i).y2 - input_boxes.at(i).y1 + 1);
    }

    vector<bool> isSuppressed(input_boxes.size(), false);
    for (int i = 0; i < int(input_boxes.size()); ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < int(input_boxes.size()); ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }
            float xx1 = (max)(input_boxes[i].x1, input_boxes[j].x1);
            float yy1 = (max)(input_boxes[i].y1, input_boxes[j].y1);
            float xx2 = (min)(input_boxes[i].x2, input_boxes[j].x2);
            float yy2 = (min)(input_boxes[i].y2, input_boxes[j].y2);

            float w = (max)(float(0), xx2 - xx1 + 1);
            float h = (max)(float(0), yy2 - yy1 + 1);
            float inter = w * h;
            float ovr = inter / (vArea[i] + vArea[j] - inter);

            if (ovr >= this->nmsThreshold)
            {
                isSuppressed[j] = true;
            }
        }
    }

    int idx_t = 0;
    input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void YOLO::detect(Mat& frame) {
    int newh = 0, neww = 0, padh = 0, padw = 0;
    Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
    this->normalize_(dstimg);

    array<int64_t, 4> input_shape_{1, 3, this->inpHeight, this->inpWidth};
    auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());

    vector<Value> ort_outputs = ort_session->Run(RunOptions{nullptr}, &input_names[0], &input_tensor_, 1, output_names.data(), output_names.size());
    const float* preds = ort_outputs[0].GetTensorMutableData<float>();

    vector<BoxInfo> generate_boxes;
    float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
    int n = 0, q = 0, i = 0, j = 0, k = 0;
    const int nout = this->num_class + 5;
    for (n = 0; n < 3; n++) {
        int num_grid_x = (int)(this->inpWidth / this->stride[n]);
        int num_grid_y = (int)(this->inpHeight / this->stride[n]);
        for (q = 0; q < 3; q++) {
            const float anchor_w = this->anchors[n][q * 2];
            const float anchor_h = this->anchors[n][q * 2 + 1];
            for (i = 0; i < num_grid_y; i++) {
                for (j = 0; j < num_grid_x; j++) {
                    float box_score = preds[4];
                    if (box_score > this->objThreshold) {
                        float cx = (preds[0] * 2.f - 0.5f + j) * this->stride[n];
                        float cy = (preds[1] * 2.f - 0.5f + i) * this->stride[n];
                        float w = powf(preds[2] * 2.f, 2.f) * anchor_w;
                        float h = powf(preds[3] * 2.f, 2.f) * anchor_h;

                        float xmin = (cx - padw - 0.5 * w) * ratiow;
                        float ymin = (cy - padh - 0.5 * h) * ratioh;
                        float xmax = (cx - padw + 0.5 * w) * ratiow;
                        float ymax = (cy - padh + 0.5 * h) * ratioh;

                        generate_boxes.push_back(BoxInfo{xmin, ymin, xmax, ymax, box_score, -1});  // -1 表示未知类别
                    }
                    preds += nout;
                }
            }
        }
    }

    nms(generate_boxes);
    for (size_t i = 0; i < generate_boxes.size(); ++i) {
        int xmin = int(generate_boxes[i].x1);
        int ymin = int(generate_boxes[i].y1);
        rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);
    }
    this->detected_boxes.clear();

    for (size_t i = 0; i < generate_boxes.size(); ++i) {
        int xmin = int(generate_boxes[i].x1);
        int ymin = int(generate_boxes[i].y1);
        rectangle(frame, Point(xmin, ymin), Point(int(generate_boxes[i].x2), int(generate_boxes[i].y2)), Scalar(0, 0, 255), 2);

        // 存储检测到的边界框信息
        this->detected_boxes.push_back(generate_boxes[i]);
    }
}

int cmpHash(const string& hash1, const string& hash2) {
    int n = 0;
    if (hash1.length() != hash2.length()) {
        return -1;
    }
    for (size_t i = 0; i < hash1.length(); ++i) {
        if (hash1[i] != hash2[i]) {
            n++;
        }
    }
    return n;
}

string dHash(const Mat& img) {
    string hash_str = "";
    Mat resized_img;
    resize(img, resized_img, Size(9, 8));
    Mat gray;
    cvtColor(resized_img, gray, COLOR_BGR2GRAY);
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            if (gray.at<uchar>(i, j) > gray.at<uchar>(i, j + 1)) {
                hash_str += "1";
            } else {
                hash_str += "0";
            }
        }
    }
    return hash_str;
}

// 提取视频差异帧序号列表
vector<int> extract_diff_frame_indices(const string& file_path, int img_dif, int img_fream) {
    VideoCapture cap(file_path);
    vector<int> diff_frame_indices;

    if (cap.isOpened()) {
        int rate = static_cast<int>(cap.get(CAP_PROP_FPS)); // 帧速率
        int FrameNumber = static_cast<int>(cap.get(CAP_PROP_FRAME_COUNT)); // 视频文件的帧数

        // 生成间隔 img_fream 帧的索引序号列表
        vector<int> frame_indices;
        for (int i = 0; i < FrameNumber; i += img_fream) {
            frame_indices.push_back(i);
        }

        string dict_img1, dict_img2;
        int sm_img;

        for (int c : frame_indices) {
            sm_img = 0;

            // 设置视频的当前帧位置
            cap.set(CAP_PROP_POS_FRAMES, c);
            Mat frame;
            cap.read(frame);

            if (frame.empty()) {
                break;
            }

            if (c % rate == 0) {
                if (!dict_img2.empty()) {
                    dict_img1 = dict_img2;
                } else {
                    dict_img1 = dHash(frame);
                }

                dict_img2 = dHash(frame);
                sm_img = cmpHash(dict_img1, dict_img2);
            }

            if (sm_img >= img_dif) {
                diff_frame_indices.push_back(c);
            }
        }

        cap.release();
    }

    return diff_frame_indices;
}

int main(int argc, char *argv[]) {

    if (argc < 5) {
        cout << "Usage: ./executable inputmp4 difference_threshold difference_frame_count outdir" << endl;
        cout << "你Example: ./executable 1.mp4 6 30 test" << endl;
        return 1;
    }

    string file_path = argv[1];
    int img_dif = atoi(argv[2]);
    int img_fream = atoi(argv[3]);
    double start_time = static_cast<double>(getTickCount());
    vector<int> diff_frame_indices = extract_diff_frame_indices(file_path, img_dif, img_fream);


    // YOLO 模型配置和初始化
    Net_config config;
    config.confThreshold = 0.5; // 置信度阈值
    config.nmsThreshold = 0.4;  // 非极大值抑制阈值
    config.objThreshold = 0.5;  // 物体置信度阈值
    config.model_path = "1.onnx";
    config.classesFile = "coco.names";

    // 初始化 YOLO 模型
    YOLO yolo_model(config);

    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <图像保存目录>" << std::endl;
        return -1;
    }

    std::string save_directory = argv[4];

    // 检查保存图像的目录是否存在，如果不存在则创建
    std::string mkdir_command = "mkdir " + save_directory;
    if (system(mkdir_command.c_str()) != 0) {
        std::cerr << "无法创建目录 '" << save_directory << "'" << std::endl;
        return -1;
    }

    // 打开视频文件
    cv::VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        std::cerr << "无法打开视频流或文件" << std::endl;
        return -1;
    }

    // 指定特定的帧列表
    int frame_count = 0;

    // 逐帧处理视频
    cv::Mat frame;
    while (cap.read(frame)) {
        frame_count++;

        // 只处理特定的帧
        if (std::find(diff_frame_indices.begin(), diff_frame_indices.end(), frame_count) == diff_frame_indices.end())
            continue;

        std::cout << "Processing frame: " << frame_count << std::endl;

        // 进行物体检测
        yolo_model.detect(frame);

        // 获取检测到的边界框信息
        std::vector<BoxInfo> boxes = yolo_model.getDetectedBoxes();
        if (!boxes.empty()) {
            // 保存带有预测框的帧到指定目录
            std::string img_name = save_directory + "/frame_" + std::to_string(frame_count) + ".jpg";
            cv::imwrite(img_name, frame);
        }
    }
    double end_time = static_cast<double>(getTickCount());
    double elapsed_time = (end_time - start_time) / getTickFrequency();
    cout << "Elapsed time: " << elapsed_time << " seconds" << endl;

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
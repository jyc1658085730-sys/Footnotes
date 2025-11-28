#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

// 基于对最大连通域 使用轮廓提取+边缘掩膜来获取车道线，使用距离变换+取样本行最大值来获取中线，并结合简单的卡尔曼滤波进行平滑处理 

//============================= 一维卡尔曼滤波器 ==============================//

class Kalman {
private:
    double x;     // 当前状态
    double p;     // 误差协方差
    double q;     // 过程噪声（调大=更信测量，更激进）
    double r;     // 测量噪声（调大=更信预测，更平滑）
    bool initialized = false;

public:
    Kalman(double process_noise, double measurement_noise) {
        q = process_noise;
        r = measurement_noise;
        p = 1.0;
    }

    double update(double measurement, int y, int h) {
        if (!initialized) {
            x = measurement;
            initialized = true;
            return x;
        }
        double far=(1-double(y)/h);
        double qf, rf;
        qf = q*(0.02 + 8*far*far);
        rf = r*(1.0 + 50*far*far*far);

        // 预测
        p = p + qf;

        // 更新
        double k = p / (p + rf);          // 卡尔曼增益
        x = x + k * (measurement - x);    // 更新状态
        p = (1 - k) * p;                  // 更新协方差

        return x;
    }
};

//============================= 巡线检测类 =================================//

class LaneDetector {
private:
    Mat cached_edge_mask;        // 缓存边缘掩膜
    Mat cached_triangle_mask;    // 缓存类三角形 ROI

    int cached_w = 0;            // 上一次图像宽
    int cached_h = 0;            // 上一次图像高

    vector<Kalman> kalman_filters;
    int n = 6;                  // 扫描间隔

public:

    // ---------- 缓存区 ----------//
    void prepareIfNeeded(int w, int h) {
        if (w == cached_w && h == cached_h &&
            !cached_edge_mask.empty() && !cached_triangle_mask.empty())
            return;

        cached_w = w;
        cached_h = h;
        
        // 初始化滤波器
        kalman_filters.clear();
        kalman_filters.resize(h/n, Kalman(2, 3));    

        // 边缘 5 像素掩膜
        cached_edge_mask = Mat::zeros(Size(w, h), CV_8UC1);
        Rect inner(5, 5, w - 10, h - 10);
        rectangle(cached_edge_mask, inner, Scalar(255), FILLED);

        // 类三角形区域
        cached_triangle_mask = Mat::zeros(Size(w, h), CV_8UC1);
        fillPoly(cached_triangle_mask, {triangle_points(w, h)}, Scalar(255));
    }

    // ---------- 预处理 ----------//
    Mat Preprocessing(const Mat& frame) {
        int w = frame.cols;
        int h = frame.rows;

        prepareIfNeeded(w, h);

        Mat gray, binary;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, gray, Size(5, 5), 0);

        threshold(gray, binary, 150, 255, THRESH_BINARY);

        // 用白色填补类三角形
        binary.setTo(255, cached_triangle_mask);

        return binary;
    }

    // ---------- 提取轮廓绘制车道 ----------//
    void extractContours(const Mat& binary, Mat& result) {
        vector<vector<Point>> contours;
        findContours(binary, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        if (contours.empty()) return;

        // 选择最大轮廓
        double max_area = 0;
        int max_idx = -1;

        for (int i = 0; i < (int)contours.size(); i++) {
            double a = contourArea(contours[i]);
            if (a > max_area) {
                max_area = a;
                max_idx = i;
            }
        }

        if (max_idx < 0 || max_area < 500) return;

        drawLines(binary, result, contours[max_idx]);
    }

    // ---------- 检测并绘制巡线 ----------//
    void drawLines(const Mat& binary, Mat& result, const vector<Point>& road_contour) {
        Mat road_mask = Mat::zeros(binary.size(), CV_8UC1);
        fillPoly(road_mask, vector<vector<Point>>{road_contour}, Scalar(255));

        vector<Point> center_points;
        int throws = 4;

        int road_y = cached_h;
        for (const Point& pt: road_contour){
            road_y = min(road_y, pt.y);
        }  

        Mat dist;
        distanceTransform(road_mask, dist, DIST_L2, 3, CV_32F);

        // 按行扫描
        for (int y = road_y; y < dist.rows-25; y += n) {

            float max_val = 0;
            int max_x = -1;
            
            const float* dist_ptr = dist.ptr<float>(y);
            
            // 在当前行找到距离变换值最大的点
            for (int x = 0; x < cached_w; x++) {
                if (dist_ptr[x] > max_val) {
                    max_val = dist_ptr[x];
                    max_x = x;
                }
            }
            
            // 只保留距离变换值足够大的点
            if (max_x != -1 && max_val > 15.0f) {
                double filtered_x = kalman_filters[(y-road_y)/n].update(max_x, y - road_y, cached_h - road_y);
                Point smooth_pt((int)(filtered_x + 0.5), y);  // 四舍五入
                if (--throws<0) center_points.push_back(smooth_pt);
            }
        }
        
        // 画线
        Mat contour_image = Mat::zeros(result.size(), result.type());
        drawContours(contour_image, vector<vector<Point>>{road_contour}, 0, Scalar(0,165,255), 2);

        for (size_t i = 1; i < center_points.size(); i++) {
            line(contour_image, center_points[i-1], center_points[i], Scalar(165,0,255), 4);
        }

        // 边缘掩膜裁剪
        Mat masked;
        contour_image.copyTo(masked, cached_edge_mask);
        result.setTo(Scalar(0,0,255), masked);
    }

    // ---------- 类三角形生成 ----------//
    vector<Point> triangle_points(int w, int h) {
        return {
            Point(w * 0.24, h-2),
            Point(w * 0.5, h * 0.7-2),
            Point(w * 0.8, h-2),
            Point(w, h-2),
            Point(w, h),
            Point(0, h),
            Point(0, h-2)
        };
    }
};

//============================== main函数 ==================================//

int main() {
    VideoCapture cap("sample2.mp4");

    if (!cap.isOpened()) {
        cout << "无法打开视频" << endl;
        return -1;
    }

    LaneDetector detector;

    Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) break;

        Mat binary = detector.Preprocessing(frame);
        
        Mat result;
        frame.copyTo(result);
        
        detector.extractContours(binary, result);
    
        imshow("Original", frame);
        imshow("Binary Result", binary);
        imshow("Lane Detection", result);

        char key = waitKey(12);
        if (key == 'q') break;
        if (key == ' ') {
            while (waitKey(0) != ' ') {}
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
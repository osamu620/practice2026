#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  cv::Mat image = cv::imread("./barbara.ppm", cv::IMREAD_ANYCOLOR);
  std::cout << "width = " << image.cols << ", ";
  std::cout << "height = " << image.rows << std::endl;

  cv::imshow("loaded image", image);

  cv::waitKey(0);
}
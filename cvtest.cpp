#include <iostream>
#include <opencv2/opencv.hpp>

auto clamp = [](int val) {
  if (val > 255) {
    val = 255;
  }
  if (val < 0) {
    val = 0;
  }
  return val;
};

cv::Mat rgb2ycbcr(cv::Mat img) {
  // RGB -> YCbCr, imgはshallowコピーなのでOK
  const int width = img.cols;
  const int height = img.rows;
  const int nc = img.channels();
  const int stride = img.step;
  uint8_t* pixel = img.data;
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      int B = pixel[i * stride + j * nc + 0];
      int G = pixel[i * stride + j * nc + 1];
      int R = pixel[i * stride + j * nc + 2];

      int Y, Cb, Cr;
      Y = .299 * R + .587 * G + .114 * B;
      Cb = -.1687 * R - .3313 * G + .5 * B;
      Cr = .5 * R - .4187 * G - .0813 * B;
      Cb += 128;
      Cr += 128;
      pixel[i * stride + j * nc + 0] = clamp(Y);
      pixel[i * stride + j * nc + 1] = clamp(Cr);
      pixel[i * stride + j * nc + 2] = clamp(Cb);
    }
  }
  return img;
}

int main() {
  cv::Mat image = cv::imread("./barbara.ppm", cv::IMREAD_ANYCOLOR);
  if (image.empty()) return EXIT_FAILURE;

  const int width = image.cols;
  const int height = image.rows;
  const int nc = image.channels();
  const int stride = image.step;
  std::cout << "width = " << width << ", ";
  std::cout << "height = " << height << std::endl;

  auto pixel = image.data;
  image = rgb2ycbcr(image);
  std::vector<cv::Mat> ycrcb;
  cv::split(image, ycrcb);  // [0] -> Y, [1] -> Cr, [2] -> Cb
  cv::resize(ycrcb[1], ycrcb[1], cv::Size(), 0.5, 0.5);  // 444 -> 420
  cv::resize(ycrcb[2], ycrcb[2], cv::Size(), 0.5, 0.5);  // 444 -> 420

  for (int y = 0; y < height; y += 8) {
    for (int x = 0; x < width; x += 8) {
      // 起点の座標(x, y)で、8x8の領域を切り出す
      cv::Mat tmp = ycrcb[0](cv::Rect(x, y, 8, 8));
      cv::Mat blk;
      tmp.convertTo(blk, CV_32F);
      blk -= 128.0f;
      // DCT
      // 量子化
      // エントロピー符号化
    }
  }
  cv::imshow("loaded image", ycrcb[0]);

  cv::waitKey(0);
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
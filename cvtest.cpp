#include <iostream>
#include <opencv2/opencv.hpp>

#include "qtable.hpp"

constexpr int FWD = 0;
constexpr int INV = 1;

template <class T>
auto clamp = [](T val) {
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
  uint8_t *pixel = img.data;
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
      pixel[i * stride + j * nc + 0] = clamp<int>(Y);
      pixel[i * stride + j * nc + 1] = clamp<int>(Cr);
      pixel[i * stride + j * nc + 2] = clamp<int>(Cb);
    }
  }
  return img;
}

template <int X>
void quantize(cv::Mat &blk, const float *qtable, float scale) {
  const int width = blk.cols;
  const int height = blk.rows;
  const int nc = blk.channels();
  const int stride = blk.step / sizeof(float);
  float *pixel = reinterpret_cast<float *>(blk.data);
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
      auto val = pixel[i * stride + j];
      auto stepsize = clamp<float>(qtable[i * stride + j] * scale);
      // 以下のif文はコンパイル時には消える
      if (X == 0)  // 量子化
        pixel[i * stride + j] = roundf(val / stepsize);
      else  // 逆量子化
        pixel[i * stride + j] = val * stepsize;
    }
  }
}

int main(int argc, char *argv[]) {
  cv::Mat image = cv::imread("./barbara.ppm", cv::IMREAD_ANYCOLOR);
  if (image.empty()) return EXIT_FAILURE;

  const int nc = image.channels();

  auto pixel = image.data;
  image = rgb2ycbcr(image);
  std::vector<cv::Mat> ycrcb;
  cv::split(image, ycrcb);  // [0] -> Y, [1] -> Cr, [2] -> Cb

  int dH = 2, dV = 2;  // 4:2:0
  cv::resize(ycrcb[1], ycrcb[1], cv::Size(), 1.0f / dH,
             1.0f / dV);  // 444 -> 420
  cv::resize(ycrcb[2], ycrcb[2], cv::Size(), 1.0f / dH,
             1.0f / dV);  // 444 -> 420

  // quality -> Qfactor -> scale
  int QF;
  int quality;
  if (argc < 2) {
    quality = 75;
  } else {
    quality = std::stoi(argv[1]);
  }
  quality = (quality == 0) ? 1 : quality;
  quality = (quality > 100) ? 100 : quality;
  if (quality <= 50) {
    QF = 5000 / quality;
  } else {
    QF = 200 - 2 * quality;
  }
  QF = (QF == 0) ? 1 : QF;
  float scale = QF / 100.0f;

  auto blkproc = [](cv::Mat &tmp, const float *qmatrix, float scale) {
    cv::Mat blk;
    tmp.convertTo(blk, CV_32F);
    blk -= 128.0f;
    // Forward DCT
    cv::dct(blk, blk, FWD);
    // 量子化
    quantize<FWD>(blk, qmatrix, scale);
    // 逆量子化
    quantize<INV>(blk, qmatrix, scale);
    // Inverse DCT
    cv::dct(blk, blk, INV);
    blk += 128.0f;
    blk.convertTo(tmp, CV_8U);
    // エントロピー符号化
  };

  // MCU(Minimum Coded Unit)単位の処理
  const int width = ycrcb[0].cols;
  const int height = ycrcb[0].rows;
  for (int y = 0, cy = 0; y < height; y += 8 * dV, cy += 8) {
    for (int x = 0, cx = 0; x < width; x += 8 * dH, cx += 8) {
      for (int i = 0; i < dV; ++i) {
        for (int j = 0; j < dH; ++j) {
          // 起点の座標(x, y)で、8x8の領域を切り出す
          cv::Mat tmpY = ycrcb[0](cv::Rect(x + j * 8, y + i * 8, 8, 8));
          blkproc(tmpY, qmatrix[0], scale);
        }
        // 起点の座標(cx, cy)で、8x8の領域を切り出す
        cv::Mat tmpCr = ycrcb[1](cv::Rect(cx, cy, 8, 8));  // Cr
        blkproc(tmpCr, qmatrix[1], scale);
        cv::Mat tmpCb = ycrcb[2](cv::Rect(cx, cy, 8, 8));  // Cb
        blkproc(tmpCb, qmatrix[1], scale);
      }
    }
  }

  cv::resize(ycrcb[1], ycrcb[1], cv::Size(), dH, dV);
  cv::resize(ycrcb[2], ycrcb[2], cv::Size(), dH, dV);
  cv::merge(ycrcb, image);
  cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);
  cv::imshow("loaded image", image);

  cv::waitKey(0);
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
#include <iostream>
#include <opencv2/opencv.hpp>

#include "bitstream.hpp"
#include "coding.hpp"
#include "jpgheaders.hpp"
#include "qtable.hpp"
#include "ycctype.hpp"

constexpr int Luma = 0;
constexpr int Chroma = 1;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("usage: cvtest.exe input output <quality> <444, 420, ...>\n");
    exit(EXIT_FAILURE);
  }
  cv::Mat image = cv::imread(argv[1], cv::IMREAD_ANYCOLOR);
  if (image.empty()) return EXIT_FAILURE;
  bitstream encbuf;

  const int nc = image.channels();

  auto pixel = image.data;
  image = rgb2ycbcr(image);
  std::vector<cv::Mat> ycrcb;
  cv::split(image, ycrcb);  // [0] -> Y, [1] -> Cr, [2] -> Cb

  int YCCtype = YUV420;
  if (argc > 4) {
    if (!strcmp("444", argv[4])) {
      YCCtype = YUV444;
    } else if (!strcmp("422", argv[4])) {
      YCCtype = YUV422;
    } else if (!strcmp("411", argv[4])) {
      YCCtype = YUV411;
    } else if (!strcmp("440", argv[4])) {
      YCCtype = YUV440;
    } else if (!strcmp("420", argv[4])) {
      YCCtype = YUV420;
    } else if (!strcmp("410", argv[4])) {
      YCCtype = YUV410;
    } else if (!strcmp("GRAY", argv[4])) {
      YCCtype = GRAY;
    } else {
      printf("Unknown YCC type\n");
      exit(EXIT_FAILURE);
    }
  }
  int dH = YCC_HV[YCCtype][0] >> 4, dV = YCC_HV[YCCtype][0] & 0x0F;
  cv::resize(ycrcb[1], ycrcb[1], cv::Size(), 1.0f / dH,
             1.0f / dV);  // 444 -> 420
  cv::resize(ycrcb[2], ycrcb[2], cv::Size(), 1.0f / dH,
             1.0f / dV);  // 444 -> 420

  const int width = ycrcb[0].cols;
  const int height = ycrcb[0].rows;
  // quality -> Qfactor -> scale
  int QF;
  int quality;
  if (argc < 4) {
    quality = 75;
  } else {
    quality = std::stoi(argv[3]);
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

  int qtable[2][64];
  for (int i = 0; i < 64; ++i) {
    if (QF != 1) {
      qtable[0][i] = static_cast<int>(clamp<float>(qmatrix[0][i] * scale));
      qtable[1][i] = static_cast<int>(clamp<float>(qmatrix[1][i] * scale));
    } else {
      qtable[0][i] = qtable[1][i] = 1;
    }
  }

  create_mainheader(width, height, nc, qtable[0], qtable[1], YCCtype, encbuf);

  // MCU(Minimum Coded Unit)単位の処理
  // prev_dc[]には前のブロックのDC成分値が入る
  int prev_dc[3] = {0};
  for (int y = 0, cy = 0; y < height; y += 8 * dV, cy += 8) {
    for (int x = 0, cx = 0; x < width; x += 8 * dH, cx += 8) {
      for (int i = 0; i < dV; ++i) {
        for (int j = 0; j < dH; ++j) {
          // 起点の座標(x, y)で、8x8の領域を切り出す
          cv::Mat tmpY = ycrcb[0](cv::Rect(x + j * 8, y + i * 8, 8, 8));
          blkproc(tmpY, qtable[Luma], prev_dc[0], Luma, encbuf);
        }
      }
      // 起点の座標(cx, cy)で、8x8の領域を切り出す
      cv::Mat tmpCr = ycrcb[2](cv::Rect(cx, cy, 8, 8));  // Cb
      blkproc(tmpCr, qtable[Chroma], prev_dc[1], Chroma, encbuf);
      cv::Mat tmpCb = ycrcb[1](cv::Rect(cx, cy, 8, 8));  // Cr
      blkproc(tmpCb, qtable[Chroma], prev_dc[2], Chroma, encbuf);
    }
  }
  size_t length = encbuf.finalize();
  std::cout << "codestream size = " << length << std::endl;
  FILE *fp = fopen(argv[2], "wb");
  if (fp == nullptr) {
    printf("Cannot open file %s\n", argv[2]);
    exit(EXIT_FAILURE);
  }
  fwrite(encbuf.get_data(), sizeof(uint8_t), length, fp);
  fclose(fp);

  // cv::resize(ycrcb[1], ycrcb[1], cv::Size(), dH, dV);
  // cv::resize(ycrcb[2], ycrcb[2], cv::Size(), dH, dV);
  // cv::merge(ycrcb, image);
  // cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);
  // cv::imshow("loaded image", image);

  // cv::waitKey(0);
  // cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
#include <iostream>
#include <opencv2/opencv.hpp>

#include "bitstream.hpp"
#include "huffman_tables.hpp"
#include "qtable.hpp"
#include "zigzag.hpp"

constexpr int FWD = 0;
constexpr int INV = 1;
constexpr int Luma = 0;
constexpr int Chroma = 1;
constexpr int DC = 0;
constexpr int AC = 1;

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

template <int X>
void encode(int run, int val, const unsigned int *t_cwd,
            const unsigned int *t_len, bitstream &encbuf) {
  int s = 0;
  int uval = (val < 0) ? -val : val;
  int bound = 1;
  while (uval >= bound) {
    bound += bound;
    s++;
  }
  if (X == 0) {
    // DC
    // t_cwd[s] を t_len[s]　ビットの値としてビットストリームに書き込む
    encbuf.put_bits(t_cwd[s], t_len[s]);
  } else {
    // AC
    // t_cwd[(run << 4) + s] を t_len[(run << 4) +
    // s]　ビットの値としてビットストリームに書き込む
    encbuf.put_bits(t_cwd[(run << 4) + s], t_len[(run << 4) + s]);
  }
  if (s != 0) {
    val = (val < 0) ? val - 1 : val;
  }
  // val を sビットの値としてビットストリームに書き込む
  encbuf.put_bits(val, s);
}

int main(int argc, char *argv[]) {
  cv::Mat image = cv::imread("./barbara.ppm", cv::IMREAD_ANYCOLOR);
  if (image.empty()) return EXIT_FAILURE;
  bitstream encbuf;

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

  auto blkproc = [](cv::Mat &tmp, const float *qmatrix, float scale,
                    int &prev_dc, int c, bitstream &encbuf) {
    cv::Mat blk;
    tmp.convertTo(blk, CV_32F);
    blk -= 128.0f;
    // Forward DCT
    cv::dct(blk, blk, FWD);
    // 量子化
    quantize<FWD>(blk, qmatrix, scale);

    auto p = reinterpret_cast<float *>(blk.data);
    // 現在のブロックのDC成分から前のブロックのDC成分を引く
    int diff = p[0] - prev_dc;
    prev_dc = p[0];  // prev_dcを更新
    encode<DC>(0, diff, DC_cwd[c], DC_len[c], encbuf);

    // AC成分のためのジグザグスキャン・ハフマン符号化
    int zero_run = 0;  // 0の連続数
    for (int i = 1; i < 64; ++i) {
      int ac = static_cast<int>(p[zigzag_scan[i]]);
      if (ac == 0) {
        zero_run++;
      } else {
        while (zero_run > 15) {
          // ZRL
          encode<AC>(0xf, 0x0, AC_cwd[c], AC_len[c], encbuf);
          zero_run -= 16;
        }
        encode<AC>(zero_run, ac, AC_cwd[c], AC_len[c], encbuf);
        zero_run = 0;
      }
    }
    if (zero_run) {
      // EOB
      encode<AC>(0x0, 0x0, AC_cwd[c], AC_len[c], encbuf);
    }

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
  // prev_dc[]には前のブロックのDC成分値が入る
  int prev_dc[3] = {0};
  for (int y = 0, cy = 0; y < height; y += 8 * dV, cy += 8) {
    for (int x = 0, cx = 0; x < width; x += 8 * dH, cx += 8) {
      for (int i = 0; i < dV; ++i) {
        for (int j = 0; j < dH; ++j) {
          // 起点の座標(x, y)で、8x8の領域を切り出す
          cv::Mat tmpY = ycrcb[0](cv::Rect(x + j * 8, y + i * 8, 8, 8));
          blkproc(tmpY, qmatrix[Luma], scale, prev_dc[0], Luma, encbuf);
        }
        // 起点の座標(cx, cy)で、8x8の領域を切り出す
        cv::Mat tmpCr = ycrcb[1](cv::Rect(cx, cy, 8, 8));  // Cr
        blkproc(tmpCr, qmatrix[Chroma], scale, prev_dc[1], Chroma, encbuf);
        cv::Mat tmpCb = ycrcb[2](cv::Rect(cx, cy, 8, 8));  // Cb
        blkproc(tmpCb, qmatrix[Chroma], scale, prev_dc[2], Chroma, encbuf);
      }
    }
  }

  size_t length = encbuf.finalize();
  std::cout << "codestream size = " << length << std::endl;

  cv::resize(ycrcb[1], ycrcb[1], cv::Size(), dH, dV);
  cv::resize(ycrcb[2], ycrcb[2], cv::Size(), dH, dV);
  cv::merge(ycrcb, image);
  cv::cvtColor(image, image, cv::COLOR_YCrCb2BGR);
  cv::imshow("loaded image", image);

  cv::waitKey(0);
  cv::destroyAllWindows();
  return EXIT_SUCCESS;
}
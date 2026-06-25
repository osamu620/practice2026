#pragma once

#include "bitstream.hpp"
#include "huffman_tables.hpp"
#include "sip.hpp"
#include "zigzag.hpp"

constexpr int FWD = 0;
constexpr int INV = 1;
constexpr int DC = 0;
constexpr int AC = 1;

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

auto blkproc = [](cv::Mat &tmp, int *qmatrix, int &prev_dc, int c,
                  bitstream &encbuf) {
  cv::Mat blk;
  tmp.convertTo(blk, CV_32F);
  blk -= 128.0f;
  // Forward DCT
  cv::dct(blk, blk, FWD);
  // 量子化
  quantize<FWD>(blk, qmatrix);

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
  quantize<INV>(blk, qmatrix);
  // Inverse DCT
  cv::dct(blk, blk, INV);
  blk += 128.0f;
  blk.convertTo(tmp, CV_8U);
  // エントロピー符号化
};
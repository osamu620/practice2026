#pragma once
#include <cstdint>
#include <vector>

class bitstream {
 private:
  uint8_t tmp;
  int32_t bits;
  std::vector<uint8_t> stream;

 public:
  // コンストラクタ: クラスの実体化の際に最初に呼ばれる
  bitstream() : tmp(0), bits(0) { stream.reserve(32768); }
  // デストラクタ: クラスの実態の寿命が切れた際に呼ばれる
  ~bitstream() {}

  inline void put_bits(uint32_t cwd, uint32_t len) {
    uint8_t b;
    while (len > 0) {
      b = static_cast<uint8_t>((cwd >> (len - 1)) & 1);
      tmp = (tmp << 1) | b;
      bits++;
      len--;
      if (bits == 8) {
        put_byte(tmp);
        if (tmp == 0xFF) {
          // バイトスタッフィング処理
          put_byte(0x00);
        }
        tmp = 0;
        bits = 0;
      }
    }
  }
  void flush() {
    tmp = tmp << (8 - bits);     // 上位へ寄せる
    tmp = tmp | (0xFF >> bits);  // 下位bitを1で埋める
    put_byte(tmp);
    if (tmp == 0xFF) {
      // バイトスタッフィング処理
      put_byte(0x00);
    }
  }

  inline void put_byte(uint8_t val) { stream.push_back(val); }

  inline void put_word(uint16_t val) {
    // Big endian
    put_byte(val >> 8);
    put_byte(val & 0xFF);
  }

  size_t finalize() {
    flush();
    return stream.size();
  }

  uint8_t* get_data() { return stream.data(); }
};
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// Basic Bloom Filter：一个字节模拟一个位，固定使用两个教学哈希
//   h1(s) = sum(c_i) mod m
//   h2(s) = sum(i * c_i) mod m        （c_i 为第 i 个字节的无符号值，i 从 1 开始）
class BloomFilter {
public:
    explicit BloomFilter(size_t m) : m_(m), bits_(m, 0) {}

    void add(const string &s) {
        bits_[h1(s)] = 1;
        bits_[h2(s)] = 1;
    }

    // true 表示 MAYBE（可能存在），false 表示 NO（一定不存在）
    bool query(const string &s) const {
        return bits_[h1(s)] != 0 && bits_[h2(s)] != 0;
    }

private:
    size_t h1(const string &s) const {
        uint64_t sum = 0;
        for (unsigned char c : s) sum = (sum + c) % m_;
        return static_cast<size_t>(sum);
    }

    size_t h2(const string &s) const {
        uint64_t sum = 0;
        uint64_t i = 1;
        for (unsigned char c : s) {
            sum = (sum + (i % m_) * c) % m_;
            ++i;
        }
        return static_cast<size_t>(sum);
    }

    size_t m_;
    vector<uint8_t> bits_;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    size_t m;
    int n;
    if (!(cin >> m >> n)) return 0;

    BloomFilter bf(m);
    for (int k = 0; k < n; ++k) {
        string op, s;
        cin >> op >> s;
        if (op == "add") {
            bf.add(s);
        } else {
            cout << (bf.query(s) ? "MAYBE" : "NO") << '\n';
        }
    }
    return 0;
}

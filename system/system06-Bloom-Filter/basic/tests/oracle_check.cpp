// 精确基线检查器：用 std::set<string> 保存真实集合，只作为答案来源，不参与 Bloom Filter 查询。
//   用法：oracle_check <input> <output>
// 检查：
//   1. 输出行数等于 query 次数，且每行只能是 NO 或 MAYBE；
//   2. 已插入的元素必须返回 MAYBE（无假阴性）；
//   3. 统计假阳性（未插入却返回 MAYBE）。
// 存在假阴性或格式错误时返回非 0。
#include <fstream>
#include <iostream>
#include <set>
#include <string>

using namespace std;

int main(int argc, char **argv) {
    if (argc != 3) {
        cerr << "usage: oracle_check <input> <output>\n";
        return 2;
    }
    ifstream in(argv[1]), out(argv[2]);
    size_t m;
    int n;
    in >> m >> n;

    set<string> truth;
    int queries = 0, inserted_hit = 0, true_neg = 0, false_pos = 0, false_neg = 0;
    for (int k = 0; k < n; ++k) {
        string op, s, ans;
        in >> op >> s;
        if (op == "add") {
            truth.insert(s);
            continue;
        }
        ++queries;
        if (!(out >> ans) || (ans != "NO" && ans != "MAYBE")) {
            cerr << "bad or missing output at query " << queries << '\n';
            return 1;
        }
        bool present = truth.count(s) > 0;
        if (present) {
            if (ans == "MAYBE") ++inserted_hit;
            else ++false_neg;
        } else {
            if (ans == "MAYBE") ++false_pos;
            else ++true_neg;
        }
    }
    string extra;
    if (out >> extra) {
        cerr << "extra output after the last query\n";
        return 1;
    }
    cout << "queries=" << queries << " inserted->MAYBE=" << inserted_hit
         << " absent->NO=" << true_neg << " false_positive=" << false_pos
         << " false_negative=" << false_neg << '\n';
    return false_neg == 0 ? 0 : 1;
}

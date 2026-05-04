#ifndef SPMV_CUDA_ARGPARSE_H
#define SPMV_CUDA_ARGPARSE_H

#include <cstdlib>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace argparse {

struct Parser {
    std::string prog;
    std::string description;

    struct Positional {
        std::string name;
        std::string help;
        std::set<std::string> choices;
        std::string *out = nullptr;
    };
    struct Option {
        std::string flag;            // e.g. "--conversion"
        std::string help;
        bool is_flag = false;        // true: bool flag; false: takes a value
        bool *bool_out = nullptr;
        int *int_out = nullptr;
    };

    std::vector<Positional> positionals;
    std::vector<Option> options;

    Parser &add_positional(std::string name, std::string help, std::string &out, std::set<std::string> choices = {}) {
        positionals.push_back({std::move(name), std::move(help), std::move(choices), &out});
        return *this;
    }
    Parser &add_flag(std::string flag, std::string help, bool &out) {
        options.push_back({std::move(flag), std::move(help), true, &out, nullptr});
        return *this;
    }
    Parser &add_int(std::string flag, std::string help, int &out) {
        options.push_back({std::move(flag), std::move(help), false, nullptr, &out});
        return *this;
    }

    [[noreturn]] void usage_and_exit(int code, const std::string &err = "") const {
        if (!err.empty()) std::cerr << "error: " << err << "\n\n";
        std::ostream &o = code == 0 ? std::cout : std::cerr;
        o << "usage: " << prog;
        for (const auto &p : positionals) o << " <" << p.name << ">";
        if (!options.empty()) o << " [options]";
        o << "\n";
        if (!description.empty()) o << "\n" << description << "\n";
        if (!positionals.empty()) {
            o << "\npositional arguments:\n";
            for (const auto &p : positionals) {
                o << "  " << p.name << "\t" << p.help;
                if (!p.choices.empty()) {
                    o << " {";
                    bool first = true;
                    for (const auto &c : p.choices) { o << (first ? "" : ","); o << c; first = false; }
                    o << "}";
                }
                o << "\n";
            }
        }
        if (!options.empty()) {
            o << "\noptions:\n";
            for (const auto &opt : options) {
                o << "  " << opt.flag;
                if (!opt.is_flag) o << " <int>";
                o << "\t" << opt.help << "\n";
            }
            o << "  -h, --help\tshow this help and exit\n";
        }
        std::exit(code);
    }

    void parse(int argc, char **argv) {
        prog = argv[0];
        std::vector<std::string> positional_vals;
        for (int i = 1; i < argc; ++i) {
            std::string a = argv[i];
            if (a == "-h" || a == "--help") usage_and_exit(0);
            if (a.rfind("--", 0) == 0) {
                bool matched = false;
                for (auto &opt : options) {
                    if (opt.flag != a) continue;
                    matched = true;
                    if (opt.is_flag) {
                        *opt.bool_out = true;
                    } else {
                        if (i + 1 >= argc) usage_and_exit(2, a + " requires a value");
                        std::string v = argv[++i];
                        try { *opt.int_out = std::stoi(v); }
                        catch (...) { usage_and_exit(2, a + " expects an integer, got '" + v + "'"); }
                    }
                    break;
                }
                if (!matched) usage_and_exit(2, "unknown option " + a);
            } else {
                positional_vals.push_back(a);
            }
        }
        if (positional_vals.size() != positionals.size()) {
            std::ostringstream oss;
            oss << "expected " << positionals.size() << " positional argument(s), got " << positional_vals.size();
            usage_and_exit(2, oss.str());
        }
        for (size_t i = 0; i < positionals.size(); ++i) {
            const auto &p = positionals[i];
            if (!p.choices.empty() && p.choices.count(positional_vals[i]) == 0) {
                usage_and_exit(2, p.name + " must be one of the listed choices, got '" + positional_vals[i] + "'");
            }
            *p.out = positional_vals[i];
        }
    }
};

} // namespace argparse

#endif // SPMV_CUDA_ARGPARSE_H

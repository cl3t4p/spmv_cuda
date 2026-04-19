#include "coo.cuh"
#include "spm_loader.cuh"
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
std::vector<std::string> split_line(const std::string &line) {
    std::istringstream iss(line);
    std::vector<std::string> result;
    std::string val;
    while (iss >> val) {
        result.push_back(val);
    }
    return result;
}

enum class Field { Real, Pattern, Integer, Complex };

enum class Symmetry { General, Symmetric, SkewSymmetric };
} // namespace

template <typename T>
bool MatrixMarketLoader<T>::load(const std::string &path, COO_Matrix<T> &out) {
    auto ifs_mtx = std::ifstream(path);
    if (!ifs_mtx.is_open()) {
        std::cerr << "file " << path << " does not exists!" << std::endl;
        return false;
    }

    std::string line;
    std::getline(ifs_mtx, line);
    auto header = split_line(line);
    if (header.size() < 5) {
        std::cerr << "malformed MatrixMarket header" << std::endl;
        return false;
    }

    Field field;
    if (const std::string &field_str = header[3]; field_str == "real") {
        field = Field::Real;
    } else if (field_str == "pattern") {
        field = Field::Pattern;
    } else if (field_str == "integer") {
        field = Field::Integer;
    } else if (field_str == "complex") {
        std::cerr << "unsupported field type: complex" << std::endl;
        return false;
    } else {
        std::cerr << "unsupported field type: " << field_str << std::endl;
        return false;
    }

    Symmetry symmetry;
    if (const std::string &sym_str = header[4]; sym_str == "general") {
        symmetry = Symmetry::General;
    } else if (sym_str == "symmetric") {
        symmetry = Symmetry::Symmetric;
    } else if (sym_str == "skew-symmetric") {
        symmetry = Symmetry::SkewSymmetric;
    } else {
        std::cerr << "unsupported symmetry: " << sym_str << std::endl;
        return false;
    }

    while (std::getline(ifs_mtx, line)) {
        if (!line.empty() && line[0] != '%') {
            break;
        }
    }

    auto coo_info = split_line(line);
    out.rows = std::stoull(coo_info[0]);
    out.cols = std::stoull(coo_info[1]);
    out.nnz = std::stoull(coo_info[2]);

    out.row_p = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * out.nnz));
    out.col_p = static_cast<uint32_t *>(malloc(sizeof(uint32_t) * out.nnz));
    out.val_p = static_cast<T *>(malloc(sizeof(T) * out.nnz));

    int index = 0;
    while (std::getline(ifs_mtx, line)) {
        auto entry_info = split_line(line);
        // MatrixMarket indices are 1-based; convert to 0-based.
        out.row_p[index] = std::stoull(entry_info[0]) - 1;
        out.col_p[index] = std::stoull(entry_info[1]) - 1;
        switch (field) {
        case Field::Real:
            out.val_p[index] = static_cast<T>(std::stod(entry_info[2]));
            break;
        case Field::Integer:
            out.val_p[index] = static_cast<T>(std::stoll(entry_info[2]));
            break;
        case Field::Pattern:
            out.val_p[index] = static_cast<T>(1);
            break;
        case Field::Complex:
            break; // unreachable: handled above
        }
        index++;
    }

    if (out.nnz != static_cast<uint32_t>(index)) {
        std::cerr << "nnz " << out.nnz << " size does not match matrix "
                  << index << "!" << std::endl;
        return false;
    }

    if (symmetry != Symmetry::General) {
        const uint32_t orig_nnz = out.nnz;
        uint32_t extra = 0;
        for (uint32_t i = 0; i < orig_nnz; i++) {
            if (out.row_p[i] != out.col_p[i]) {
                extra++;
            }
        }
        const uint32_t new_nnz = orig_nnz + extra;

        // Because the matrix is symmetric then we need to copy the bottom
        // triangle
        out.row_p = static_cast<uint32_t *>(
            realloc(out.row_p, sizeof(uint32_t) * new_nnz));
        out.col_p = static_cast<uint32_t *>(
            realloc(out.col_p, sizeof(uint32_t) * new_nnz));
        out.val_p = static_cast<T *>(realloc(out.val_p, sizeof(T) * new_nnz));

        uint32_t idx = orig_nnz;
        for (uint32_t i = 0; i < orig_nnz; i++) {
            if (out.row_p[i] != out.col_p[i]) {
                out.row_p[idx] = out.col_p[i];
                out.col_p[idx] = out.row_p[i];
                out.val_p[idx] = (symmetry == Symmetry::SkewSymmetric)
                                     ? static_cast<T>(-out.val_p[i])
                                     : out.val_p[i];
                idx++;
            }
        }
        out.nnz = new_nnz;
    }

    return true;
}

// Explicit instantiations — must match the types the static_assert in COO<T>
// allows.
template class MatrixMarketLoader<int>;
template class MatrixMarketLoader<float>;
template class MatrixMarketLoader<double>;

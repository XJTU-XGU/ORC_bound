#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

double now_seconds() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    using clock = std::chrono::steady_clock;
    static const auto start = clock::now();
    const auto elapsed = clock::now() - start;
    return std::chrono::duration<double>(elapsed).count();
#endif
}

struct CandidatePair {
    py::ssize_t left;
    py::ssize_t right;
};

struct ResidualScratch {
    std::vector<double> left_mass;
    std::vector<double> right_mass;
    std::vector<py::ssize_t> active_left;
    std::vector<py::ssize_t> active_right;
    std::vector<std::vector<CandidatePair>> candidate_buckets;

    explicit ResidualScratch(int l_shell)
        : candidate_buckets(static_cast<std::size_t>(l_shell + 1)) {}

    void reset(py::ssize_t x_size, py::ssize_t y_size, int l_shell) {
        left_mass.resize(static_cast<std::size_t>(x_size));
        right_mass.resize(static_cast<std::size_t>(y_size));
        active_left.clear();
        active_right.clear();
        const auto bucket_count = static_cast<std::size_t>(l_shell + 1);
        if (candidate_buckets.size() != bucket_count) {
            candidate_buckets.resize(bucket_count);
        }
        for (auto &bucket : candidate_buckets) {
            bucket.clear();
        }
    }
};

struct FastScratch {
    std::unordered_map<std::int64_t, double> left;
    std::unordered_map<std::int64_t, double> right;

    void reset(std::int64_t left_size, std::int64_t right_size) {
        left.clear();
        right.clear();
        const auto left_reserve = static_cast<std::size_t>(2 * left_size + 1);
        const auto right_reserve = static_cast<std::size_t>(2 * right_size + 1);
        if (left.bucket_count() < left_reserve) {
            left.reserve(left_reserve);
        }
        if (right.bucket_count() < right_reserve) {
            right.reserve(right_reserve);
        }
    }
};

void consume_candidate(
    const CandidatePair &candidate,
    std::vector<double> &left_mass,
    std::vector<double> &right_mass,
    double shell_distance,
    double tol,
    double &upper_bound
) {
    double &left = left_mass[static_cast<std::size_t>(candidate.left)];
    double &right = right_mass[static_cast<std::size_t>(candidate.right)];
    const double delta = std::min(left, right);
    if (delta > tol) {
        left -= delta;
        right -= delta;
        upper_bound += shell_distance * delta;
    }
}

void require_1d(const py::buffer_info &info, const char *name) {
    if (info.ndim != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1-D array");
    }
}

void validate_inputs(
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &edges,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &measure_indptr,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &measure_indices,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &measure_values,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &distances,
    int l_shell,
    const std::string &rbar_mode,
    double tol
) {
    const auto edge_info = edges.request();
    const auto indptr_info = measure_indptr.request();
    const auto indices_info = measure_indices.request();
    const auto values_info = measure_values.request();
    const auto distance_info = distances.request();

    if (edge_info.ndim != 2 || edge_info.shape[1] != 2) {
        throw std::invalid_argument("edges must have shape (num_edges, 2)");
    }
    require_1d(indptr_info, "measure_indptr");
    require_1d(indices_info, "measure_indices");
    require_1d(values_info, "measure_values");
    if (distance_info.ndim != 2 || distance_info.shape[0] != distance_info.shape[1]) {
        throw std::invalid_argument("distances must be a square 2-D array");
    }
    if (indices_info.shape[0] != values_info.shape[0]) {
        throw std::invalid_argument("measure_indices and measure_values must have same length");
    }
    if (l_shell < 0) {
        throw std::invalid_argument("l_shell must be non-negative");
    }
    if (rbar_mode != "local-max" && rbar_mode != "global-diam") {
        throw std::invalid_argument("rbar_mode must be 'local-max' or 'global-diam'");
    }
    if (tol < 0.0) {
        throw std::invalid_argument("tol must be non-negative");
    }

    const py::ssize_t n = distance_info.shape[0];
    const py::ssize_t nnz = indices_info.shape[0];
    if (indptr_info.shape[0] != n + 1) {
        throw std::invalid_argument("measure_indptr length must be distances.shape[0] + 1");
    }

    auto e = edges.unchecked<2>();
    auto indptr = measure_indptr.unchecked<1>();
    auto indices = measure_indices.unchecked<1>();
    auto values = measure_values.unchecked<1>();
    auto D = distances.unchecked<2>();

    for (py::ssize_t row = 0; row < n; ++row) {
        const std::int64_t start = indptr(row);
        const std::int64_t end = indptr(row + 1);
        if (start < 0 || end < start || end > nnz) {
            throw std::invalid_argument("measure_indptr must be monotone and within nnz");
        }
        for (std::int64_t p = start; p < end; ++p) {
            const std::int64_t col = indices(p);
            if (col < 0 || col >= n) {
                throw std::invalid_argument("measure_indices contains an out-of-range node index");
            }
            if (!std::isfinite(values(p)) || values(p) < -tol) {
                throw std::invalid_argument("measure_values must be finite and non-negative");
            }
        }
    }

    const py::ssize_t num_edges = edge_info.shape[0];
    for (py::ssize_t i = 0; i < num_edges; ++i) {
        const std::int64_t x = e(i, 0);
        const std::int64_t y = e(i, 1);
        if (x < 0 || x >= n || y < 0 || y >= n) {
            throw std::invalid_argument("edges contains an out-of-range node index");
        }
        const double edge_distance = D(x, y);
        if (!std::isfinite(edge_distance) || edge_distance <= 0.0) {
            throw std::invalid_argument("distances[x, y] must be finite and positive for every edge");
        }
    }
}

void validate_fast_inputs(
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &edges,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &measure_indptr,
    const py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> &measure_indices,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &measure_values,
    double tol
) {
    const auto edge_info = edges.request();
    const auto indptr_info = measure_indptr.request();
    const auto indices_info = measure_indices.request();
    const auto values_info = measure_values.request();

    if (edge_info.ndim != 2 || edge_info.shape[1] != 2) {
        throw std::invalid_argument("edges must have shape (num_edges, 2)");
    }
    require_1d(indptr_info, "measure_indptr");
    require_1d(indices_info, "measure_indices");
    require_1d(values_info, "measure_values");
    if (indices_info.shape[0] != values_info.shape[0]) {
        throw std::invalid_argument("measure_indices and measure_values must have same length");
    }
    if (indptr_info.shape[0] < 2) {
        throw std::invalid_argument("measure_indptr must contain at least two entries");
    }
    if (tol < 0.0) {
        throw std::invalid_argument("tol must be non-negative");
    }

    const py::ssize_t n = indptr_info.shape[0] - 1;
    const py::ssize_t nnz = indices_info.shape[0];
    const auto edge_view = edges.unchecked<2>();
    const auto indptr = measure_indptr.unchecked<1>();
    const auto indices = measure_indices.unchecked<1>();
    const auto values = measure_values.unchecked<1>();

    if (indptr(0) != 0 || indptr(n) != nnz) {
        throw std::invalid_argument("measure_indptr must start at 0 and end at nnz");
    }
    for (py::ssize_t row = 0; row < n; ++row) {
        const std::int64_t start = indptr(row);
        const std::int64_t end = indptr(row + 1);
        if (start < 0 || end < start || end > nnz) {
            throw std::invalid_argument("measure_indptr must be monotone and within nnz");
        }
        for (std::int64_t p = start; p < end; ++p) {
            const std::int64_t col = indices(p);
            if (col < 0 || col >= n) {
                throw std::invalid_argument("measure_indices contains an out-of-range node index");
            }
            if (!std::isfinite(values(p)) || values(p) < -tol) {
                throw std::invalid_argument("measure_values must be finite and non-negative");
            }
        }
    }

    for (py::ssize_t edge_id = 0; edge_id < edge_info.shape[0]; ++edge_id) {
        const std::int64_t x = edge_view(edge_id, 0);
        const std::int64_t y = edge_view(edge_id, 1);
        if (x < 0 || x >= n || y < 0 || y >= n) {
            throw std::invalid_argument("edges contains an out-of-range node index");
        }
        if (x == y) {
            throw std::invalid_argument("fast Algorithm 1 requires edges with distinct endpoints");
        }
    }
}

double compute_global_diameter(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &distances
) {
    const auto D = distances.unchecked<2>();
    const py::ssize_t n = D.shape(0);
    double diameter = 0.0;
    for (py::ssize_t i = 0; i < n; ++i) {
        for (py::ssize_t j = 0; j < n; ++j) {
            const double d = D(i, j);
            if (std::isfinite(d) && d > diameter) {
                diameter = d;
            }
        }
    }
    return diameter;
}

}  // namespace

py::dict compute_residual_shell(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> edges,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> measure_indptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> measure_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> measure_values,
    py::array_t<double, py::array::c_style | py::array::forcecast> distances,
    int l_shell,
    const std::string &rbar_mode,
    double tol,
    int num_threads,
    int progress_interval
) {
    validate_inputs(
        edges,
        measure_indptr,
        measure_indices,
        measure_values,
        distances,
        l_shell,
        rbar_mode,
        tol
    );

    const auto edge_info = edges.request();
    const py::ssize_t num_edges = edge_info.shape[0];

    auto w1_upper_bounds = py::array_t<double>(num_edges);
    auto curvatures = py::array_t<double>(num_edges);
    auto edge_time_seconds = py::array_t<double>(num_edges);

    const double global_diameter =
        (rbar_mode == "global-diam") ? compute_global_diameter(distances) : 0.0;

#ifdef _OPENMP
    const int actual_threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
#else
    const int actual_threads = 1;
    (void)num_threads;
#endif

    const double total_started_at = now_seconds();
    const std::int64_t progress_step =
        (progress_interval > 0) ? static_cast<std::int64_t>(progress_interval) : 0;
    std::atomic<std::int64_t> completed_edges{0};

    {
        py::gil_scoped_release release;

        const auto edge_view = edges.unchecked<2>();
        const auto indptr = measure_indptr.unchecked<1>();
        const auto indices = measure_indices.unchecked<1>();
        const auto values = measure_values.unchecked<1>();
        const auto D = distances.unchecked<2>();
        auto w1_out = w1_upper_bounds.mutable_unchecked<1>();
        auto kappa_out = curvatures.mutable_unchecked<1>();
        auto time_out = edge_time_seconds.mutable_unchecked<1>();

        const auto compute_one_edge = [&](py::ssize_t edge_id, ResidualScratch &scratch) {
            const double edge_started_at = now_seconds();

            const std::int64_t x = edge_view(edge_id, 0);
            const std::int64_t y = edge_view(edge_id, 1);

            const std::int64_t x_start = indptr(x);
            const std::int64_t x_end = indptr(x + 1);
            const std::int64_t y_start = indptr(y);
            const std::int64_t y_end = indptr(y + 1);

            const py::ssize_t x_size = static_cast<py::ssize_t>(x_end - x_start);
            const py::ssize_t y_size = static_cast<py::ssize_t>(y_end - y_start);

            scratch.reset(x_size, y_size, l_shell);

            for (py::ssize_t i = 0; i < x_size; ++i) {
                scratch.left_mass[static_cast<std::size_t>(i)] =
                    std::max(0.0, values(x_start + i));
            }
            for (py::ssize_t j = 0; j < y_size; ++j) {
                scratch.right_mass[static_cast<std::size_t>(j)] =
                    std::max(0.0, values(y_start + j));
            }

            for (py::ssize_t i = 0; i < x_size; ++i) {
                const std::int64_t u = indices(x_start + i);
                for (py::ssize_t j = 0; j < y_size; ++j) {
                    const std::int64_t v = indices(y_start + j);
                    const double d = D(u, v);
                    if (!std::isfinite(d) || d > static_cast<double>(l_shell) + tol) {
                        continue;
                    }

                    int bucket = static_cast<int>(d);
                    if (bucket >= 0 && bucket <= l_shell) {
                        scratch.candidate_buckets[static_cast<std::size_t>(bucket)]
                            .push_back({i, j});
                    }
                }
            }

            double ub = 0.0;
            for (
                std::size_t shell = 0;
                shell < scratch.candidate_buckets.size();
                ++shell
            ) {
                const auto &bucket = scratch.candidate_buckets[shell];
                const double shell_distance = static_cast<double>(shell);
                for (const CandidatePair &candidate : bucket) {
                    consume_candidate(
                        candidate,
                        scratch.left_mass,
                        scratch.right_mass,
                        shell_distance,
                        tol,
                        ub
                    );
                }
            }

            double residual_mass = 0.0;
            scratch.active_left.clear();
            scratch.active_right.clear();
            for (py::ssize_t i = 0; i < x_size; ++i) {
                const double mass = scratch.left_mass[static_cast<std::size_t>(i)];
                residual_mass += mass;
                if (mass > tol) {
                    scratch.active_left.push_back(i);
                }
            }
            for (py::ssize_t j = 0; j < y_size; ++j) {
                if (scratch.right_mass[static_cast<std::size_t>(j)] > tol) {
                    scratch.active_right.push_back(j);
                }
            }

            double rbar = 0.0;
            if (residual_mass > tol) {
                if (rbar_mode == "local-max") {
                    for (py::ssize_t i : scratch.active_left) {
                        const std::int64_t u = indices(x_start + i);
                        for (py::ssize_t j : scratch.active_right) {
                            const std::int64_t v = indices(y_start + j);
                            const double d = D(u, v);
                            if (std::isfinite(d) && d > rbar) {
                                rbar = d;
                            }
                        }
                    }
                } else {
                    rbar = global_diameter;
                }
            }

            ub += rbar * residual_mass;

            const double edge_distance = D(x, y);
            w1_out(edge_id) = ub;
            kappa_out(edge_id) = 1.0 - (ub / edge_distance);
            time_out(edge_id) = now_seconds() - edge_started_at;

            if (progress_step > 0) {
                const std::int64_t completed = completed_edges.fetch_add(1) + 1;
                if (
                    completed == static_cast<std::int64_t>(num_edges) ||
                    completed % progress_step == 0
                ) {
#ifdef _OPENMP
#pragma omp critical(residual_shell_progress)
#endif
                    {
                        std::cout
                            << "  residual_shell: "
                            << completed << "/"
                            << static_cast<std::int64_t>(num_edges)
                            << " edges" << std::endl;
                    }
                }
            }
        };

#ifdef _OPENMP
#pragma omp parallel num_threads(actual_threads)
        {
            ResidualScratch scratch(l_shell);
#pragma omp for schedule(dynamic, 16)
            for (py::ssize_t edge_id = 0; edge_id < num_edges; ++edge_id) {
                compute_one_edge(edge_id, scratch);
            }
        }
#else
        ResidualScratch scratch(l_shell);
        for (py::ssize_t edge_id = 0; edge_id < num_edges; ++edge_id) {
            compute_one_edge(edge_id, scratch);
        }
#endif
    }

    const double elapsed = now_seconds() - total_started_at;

    py::dict result;
    result["w1_upper_bounds"] = w1_upper_bounds;
    result["curvatures"] = curvatures;
    result["edge_time_seconds"] = edge_time_seconds;
    result["elapsed_seconds"] = elapsed;
    result["sum_edge_time_seconds"] =
        py::module_::import("numpy").attr("sum")(edge_time_seconds).cast<double>();
    result["num_threads"] = actual_threads;
    return result;
}

py::dict compute_fast_algo1(
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> edges,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> measure_indptr,
    py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> measure_indices,
    py::array_t<double, py::array::c_style | py::array::forcecast> measure_values,
    double tol,
    int num_threads
) {
    validate_fast_inputs(
        edges,
        measure_indptr,
        measure_indices,
        measure_values,
        tol
    );

    const py::ssize_t num_edges = edges.request().shape[0];
    auto w1_upper_bounds = py::array_t<double>(num_edges);
    auto curvatures = py::array_t<double>(num_edges);
    auto edge_time_seconds = py::array_t<double>(num_edges);

#ifdef _OPENMP
    const int actual_threads = (num_threads > 0) ? num_threads : omp_get_max_threads();
#else
    const int actual_threads = 1;
    (void)num_threads;
#endif

    const double total_started_at = now_seconds();

    {
        py::gil_scoped_release release;

        const auto edge_view = edges.unchecked<2>();
        const auto indptr = measure_indptr.unchecked<1>();
        const auto indices = measure_indices.unchecked<1>();
        const auto values = measure_values.unchecked<1>();
        auto w1_out = w1_upper_bounds.mutable_unchecked<1>();
        auto kappa_out = curvatures.mutable_unchecked<1>();
        auto time_out = edge_time_seconds.mutable_unchecked<1>();

        const auto compute_one_edge = [&](py::ssize_t edge_id, FastScratch &scratch) {
            const double edge_started_at = now_seconds();
            const std::int64_t x = edge_view(edge_id, 0);
            const std::int64_t y = edge_view(edge_id, 1);

            const std::int64_t x_start = indptr(x);
            const std::int64_t x_end = indptr(x + 1);
            const std::int64_t y_start = indptr(y);
            const std::int64_t y_end = indptr(y + 1);

            scratch.reset(x_end - x_start, y_end - y_start);
            auto &left = scratch.left;
            auto &right = scratch.right;

            for (std::int64_t p = x_start; p < x_end; ++p) {
                left[indices(p)] += values(p);
            }
            for (std::int64_t p = y_start; p < y_end; ++p) {
                right[indices(p)] += values(p);
            }

            const auto mass_at = [](
                const std::unordered_map<std::int64_t, double> &measure,
                std::int64_t node
            ) {
                const auto found = measure.find(node);
                return (found == measure.end()) ? 0.0 : found->second;
            };

            const double alpha0 = std::min(mass_at(left, x), mass_at(right, y));
            const double denominator = 1.0 - alpha0;
            double upper_bound = alpha0;

            if (denominator > tol) {
                const double inv = 1.0 / denominator;
                const double nu_xy =
                    (mass_at(left, y) - ((y == x) ? alpha0 : 0.0)) * inv;
                const double nu_yx =
                    (mass_at(right, x) - ((x == y) ? alpha0 : 0.0)) * inv;

                double max_sum = 0.0;
                double min_sum = 0.0;
                const auto &smaller = (left.size() <= right.size()) ? left : right;
                const auto &larger = (left.size() <= right.size()) ? right : left;
                const bool smaller_is_left = &smaller == &left;

                for (const auto &entry : smaller) {
                    const std::int64_t z = entry.first;
                    if (z == x || z == y) {
                        continue;
                    }
                    const auto other = larger.find(z);
                    if (other == larger.end()) {
                        continue;
                    }

                    double vx = (smaller_is_left ? entry.second : other->second) * inv;
                    double vy = (smaller_is_left ? other->second : entry.second) * inv;
                    if (vx < 0.0 && vx > -tol) {
                        vx = 0.0;
                    }
                    if (vy < 0.0 && vy > -tol) {
                        vy = 0.0;
                    }
                    max_sum += std::max(vx, vy);
                    min_sum += std::min(vx, vy);
                }

                double inner = 1.0 - min_sum;
                const double t1 = 1.0 - nu_xy - nu_yx - max_sum;
                if (t1 > 0.0) {
                    inner += t1;
                }
                const double t2 = 1.0 - nu_xy - nu_yx - min_sum;
                if (t2 > 0.0) {
                    inner += t2;
                }
                upper_bound = std::max(
                    0.0,
                    alpha0 + (1.0 - alpha0) * inner
                );
            }

            w1_out(edge_id) = upper_bound;
            kappa_out(edge_id) = 1.0 - upper_bound;
            time_out(edge_id) = now_seconds() - edge_started_at;
        };

#ifdef _OPENMP
#pragma omp parallel num_threads(actual_threads)
        {
            FastScratch scratch;
#pragma omp for schedule(dynamic, 16)
            for (py::ssize_t edge_id = 0; edge_id < num_edges; ++edge_id) {
                compute_one_edge(edge_id, scratch);
            }
        }
#else
        FastScratch scratch;
        for (py::ssize_t edge_id = 0; edge_id < num_edges; ++edge_id) {
            compute_one_edge(edge_id, scratch);
        }
#endif
    }

    const double elapsed = now_seconds() - total_started_at;
    py::dict result;
    result["w1_upper_bounds"] = w1_upper_bounds;
    result["curvatures"] = curvatures;
    result["edge_time_seconds"] = edge_time_seconds;
    result["elapsed_seconds"] = elapsed;
    result["sum_edge_time_seconds"] =
        py::module_::import("numpy").attr("sum")(edge_time_seconds).cast<double>();
    result["num_threads"] = actual_threads;
    return result;
}

PYBIND11_MODULE(_residual_shell_cpp, m) {
    m.doc() = "OpenMP curvature approximation kernels";
    m.def(
        "compute_residual_shell",
        &compute_residual_shell,
        py::arg("edges"),
        py::arg("measure_indptr"),
        py::arg("measure_indices"),
        py::arg("measure_values"),
        py::arg("distances"),
        py::arg("l_shell") = 2,
        py::arg("rbar_mode") = "local-max",
        py::arg("tol") = 1e-12,
        py::arg("num_threads") = 0,
        py::arg("progress_interval") = 0,
        R"pbdoc(
Compute residual-shell W1 upper bounds and curvature values for many edges.

All graph nodes must already be represented by contiguous integer indices
0..n-1. The measure arrays use CSR layout:
measure_indices[measure_indptr[x]:measure_indptr[x+1]] are support nodes for x,
and measure_values over the same range are their masses.
Set progress_interval > 0 to print a progress line every N completed edges.
)pbdoc"
    );
    m.def(
        "compute_fast_algo1",
        &compute_fast_algo1,
        py::arg("edges"),
        py::arg("measure_indptr"),
        py::arg("measure_indices"),
        py::arg("measure_values"),
        py::arg("tol") = 1e-12,
        py::arg("num_threads") = 0,
        R"pbdoc(
Compute corrected fast Algorithm 1 W1 upper bounds for many unweighted edges.

The implementation matches algorithms.fast_upper_bound_algo1: it sums over
the common measure support excluding the edge endpoints. Curvature is returned
as 1 - W1 because each input edge has unweighted hop distance one.
)pbdoc"
    );
}


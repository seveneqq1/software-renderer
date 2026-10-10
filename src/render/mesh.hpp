#pragma once

#include "../math/vec_mat.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace sr {

struct Vertex {
    Vec3 p;
    Vec3 n;
    Vec2 uv;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    int triangle_count() const { return static_cast<int>(indices.size() / 3); }
};

inline void mesh_compute_smooth_normals(Mesh& mesh) {
    for (auto& v : mesh.vertices) v.n = {0.f, 0.f, 0.f};
    for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        Vertex& a = mesh.vertices[mesh.indices[i]];
        Vertex& b = mesh.vertices[mesh.indices[i + 1]];
        Vertex& c = mesh.vertices[mesh.indices[i + 2]];
        const Vec3 fn = cross(b.p - a.p, c.p - a.p);
        a.n += fn;
        b.n += fn;
        c.n += fn;
    }
    for (auto& v : mesh.vertices) v.n = normalize(v.n);
}

inline Mesh make_plane(float size, int segments) {
    Mesh m;
    segments = std::max(segments, 1);
    for (int y = 0; y <= segments; ++y) {
        for (int x = 0; x <= segments; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(segments);
            const float v = static_cast<float>(y) / static_cast<float>(segments);
            Vertex vert;
            vert.p = {(u - 0.5f) * size, 0.f, (v - 0.5f) * size};
            vert.n = {0.f, 1.f, 0.f};
            vert.uv = {u * 4.f, v * 4.f};
            m.vertices.push_back(vert);
        }
    }
    const int stride = segments + 1;
    for (int y = 0; y < segments; ++y) {
        for (int x = 0; x < segments; ++x) {
            const uint32_t a = static_cast<uint32_t>(y * stride + x);
            const uint32_t b = a + 1;
            const uint32_t c = a + static_cast<uint32_t>(stride);
            const uint32_t d = c + 1;
            m.indices.insert(m.indices.end(), {a, b, c, b, d, c});
        }
    }
    return m;
}

inline Mesh make_uv_sphere(float radius, int slices, int stacks) {
    Mesh m;
    slices = std::max(slices, 3);
    stacks = std::max(stacks, 2);
    for (int i = 0; i <= stacks; ++i) {
        const float v = static_cast<float>(i) / static_cast<float>(stacks);
        const float phi = v * kPi;
        const float sp = std::sin(phi), cp = std::cos(phi);
        for (int j = 0; j <= slices; ++j) {
            const float u = static_cast<float>(j) / static_cast<float>(slices);
            const float th = u * kTau;
            const float st = std::sin(th), ct = std::cos(th);
            Vertex vert;
            vert.n = {st * sp, cp, ct * sp};
            vert.p = vert.n * radius;
            vert.uv = {u, 1.f - v};
            m.vertices.push_back(vert);
        }
    }
    const int stride = slices + 1;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            const uint32_t a = static_cast<uint32_t>(i * stride + j);
            const uint32_t b = a + static_cast<uint32_t>(stride);
            const uint32_t c = a + 1;
            const uint32_t d = b + 1;
            m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
        }
    }
    return m;
}

inline Mesh make_torus(float major_r, float minor_r, int seg, int sides) {
    Mesh m;
    seg = std::max(seg, 3);
    sides = std::max(sides, 3);
    for (int i = 0; i <= seg; ++i) {
        const float u = static_cast<float>(i) / static_cast<float>(seg);
        const float th = u * kTau;
        const float ct = std::cos(th), st = std::sin(th);
        for (int j = 0; j <= sides; ++j) {
            const float v = static_cast<float>(j) / static_cast<float>(sides);
            const float ph = v * kTau;
            const float cp = std::cos(ph), sp = std::sin(ph);
            Vertex vert;
            vert.n = {cp * ct, sp, cp * st};
            vert.p = { (major_r + minor_r * cp) * ct, minor_r * sp, (major_r + minor_r * cp) * st };
            vert.uv = {u * 2.f, v};
            m.vertices.push_back(vert);
        }
    }
    const int stride = sides + 1;
    for (int i = 0; i < seg; ++i) {
        for (int j = 0; j < sides; ++j) {
            const uint32_t a = static_cast<uint32_t>(i * stride + j);
            const uint32_t b = a + static_cast<uint32_t>(stride);
            const uint32_t c = a + 1;
            const uint32_t d = b + 1;
            m.indices.insert(m.indices.end(), {a, c, b, b, c, d});
        }
    }
    return m;
}

inline bool load_obj(const std::string& path, Mesh& out, std::string* err = nullptr) {
    std::ifstream in(path);
    if (!in) {
        if (err) *err = "cannot open " + path;
        return false;
    }

    std::vector<Vec3> pos;
    std::vector<Vec2> uvs;
    std::vector<Vec3> nrm;
    struct Key {
        int p = 0, t = 0, n = 0;
        bool operator==(const Key& o) const { return p == o.p && t == o.t && n == o.n; }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return static_cast<size_t>(k.p) * 73856093u ^ static_cast<size_t>(k.t) * 19349663u ^
                   static_cast<size_t>(k.n) * 83492791u;
        }
    };
    std::unordered_map<Key, uint32_t, KeyHash> map;
    Mesh mesh;

    auto resolve = [](int idx, int count) -> int {
        if (idx > 0) return idx - 1;
        if (idx < 0) return count + idx;
        return -1;
    };

    auto corner = [&](const std::string& tok) -> uint32_t {
        Key key;
        int slashes = 0;
        for (char c : tok)
            if (c == '/') ++slashes;
        if (slashes == 0) {
            key.p = std::stoi(tok);
        } else if (slashes == 1) {
            const auto a = tok.find('/');
            key.p = std::stoi(tok.substr(0, a));
            key.t = std::stoi(tok.substr(a + 1));
        } else {
            const auto a = tok.find('/');
            const auto b = tok.find('/', a + 1);
            key.p = std::stoi(tok.substr(0, a));
            if (b > a + 1) key.t = std::stoi(tok.substr(a + 1, b - a - 1));
            if (b + 1 < tok.size()) key.n = std::stoi(tok.substr(b + 1));
        }
        key.p = resolve(key.p, static_cast<int>(pos.size()));
        key.t = (key.t == 0) ? -1 : resolve(key.t, static_cast<int>(uvs.size()));
        key.n = (key.n == 0) ? -1 : resolve(key.n, static_cast<int>(nrm.size()));
        const auto it = map.find(key);
        if (it != map.end()) return it->second;
        Vertex v;
        v.p = (key.p >= 0 && key.p < static_cast<int>(pos.size())) ? pos[static_cast<size_t>(key.p)] : Vec3{};
        v.uv = (key.t >= 0 && key.t < static_cast<int>(uvs.size())) ? uvs[static_cast<size_t>(key.t)] : Vec2{};
        v.n = (key.n >= 0 && key.n < static_cast<int>(nrm.size())) ? nrm[static_cast<size_t>(key.n)] : Vec3{0, 1, 0};
        const uint32_t id = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(v);
        map.emplace(key, id);
        return id;
    };

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tag;
        ss >> tag;
        if (tag == "v") {
            Vec3 p;
            ss >> p.x >> p.y >> p.z;
            pos.push_back(p);
        } else if (tag == "vt") {
            Vec2 t;
            ss >> t.x >> t.y;
            uvs.push_back(t);
        } else if (tag == "vn") {
            Vec3 n;
            ss >> n.x >> n.y >> n.z;
            nrm.push_back(n);
        } else if (tag == "f") {
            std::vector<uint32_t> face;
            std::string tok;
            while (ss >> tok) face.push_back(corner(tok));
            for (size_t i = 1; i + 1 < face.size(); ++i) {
                mesh.indices.push_back(face[0]);
                mesh.indices.push_back(face[i]);
                mesh.indices.push_back(face[i + 1]);
            }
        }
    }

    bool need_n = mesh.vertices.empty();
    for (const auto& v : mesh.vertices) {
        if (length_sq(v.n) < 1e-10f) {
            need_n = true;
            break;
        }
    }
    if (need_n) mesh_compute_smooth_normals(mesh);
    out = std::move(mesh);
    return !out.indices.empty();
}

}  // namespace sr

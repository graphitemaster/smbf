//Copyright (c) 2015 Dale Weiler
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <assert.h>
#include <stdarg.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// Utilities
#include <vector>
#include <string>
#include <unordered_map>
#include <utility>

#undef min
#undef max
#undef clamp

namespace u {

template <typename T>
using vector = std::vector<T>;
using string = std::string;

template <typename T1, typename T2>
using map = std::unordered_map<T1, T2>;

template <typename T>
typename std::remove_reference<T>::type&& move(T&& t) {
    return std::move(t);
}

// Little search implementation
template <typename I1, typename I2>
I1 search(I1 first1, I1 last1, I2 first2, I2 last2) {
    for (; ; ++first1) {
        I1 it1 = first1;
        for (I2 it2 = first2; ; ++it1, ++it2) {
            if (it2 == last2)
                return first1;
            if (it1 == last1)
                return last1;
            if (!(*it1 == *it2))
                break;
        }
    }
}

// Replace all occurrences of a string with a different string in a string
static void replace_all(string &s, const char *before, const char *after) {
    const size_t beforeLength = strlen(before);
    string result;
    string::const_iterator end_ = s.end();
    string::const_iterator current = s.begin();
    string::const_iterator next = search(current, end_, before, before + beforeLength);
    while (next != end_) {
        result.append(current, next);
        result.append(after);
        current = next + beforeLength;
        next = search(current, end_, before, before + beforeLength);
    }
    result.append(current, next);
    s.swap(result);
}

// An implementation of swap
template <typename T>
inline void swap(T &lhs, T &rhs) {
    T temp = move(rhs);
    rhs = move(lhs);
    lhs = move(temp);
}

// A small implementation of boost.optional
namespace detail {
    template <typename T, bool>
    union optional_cast;

    template <typename T>
    union optional_cast<T, true> {
        const void *p;
        const T *data;
    };

    template <typename T>
    union optional_cast<T, false> {
        void *p;
        T *data;
    };
}

struct optional_none { };

typedef int optional_none::*none_t;
none_t const none = none_t(0);

template <typename T>
struct optional {
    optional();
    optional(none_t);
    optional(const T &value);
    optional(const optional<T> &opt);

    ~optional();

    optional &operator=(const optional<T> &opt);

    operator bool() const;
    T &operator *();
    const T &operator*() const;

private:
    void *storage();
    const void *storage() const;
    T &get();
    const T &get() const;
    void destruct();
    void construct(const T &data);

    bool m_init;
    alignas(alignof(T)) unsigned char m_data[sizeof(T)];
};

template <typename T>
optional<T>::optional()
    : m_init(false)
{
}

template <typename T>
optional<T>::optional(none_t)
    : m_init(false)
{
}

template <typename T>
optional<T>::optional(const T &value)
    : m_init(true)
{
    construct(value);
}

template <typename T>
optional<T>::optional(const optional<T> &opt)
    : m_init(opt.m_init)
{
    if (m_init)
        construct(opt.get());
}

template <typename T>
optional<T>::~optional() {
    destruct();
}

template <typename T>
optional<T> &optional<T>::operator=(const optional<T> &opt) {
    destruct();
    if ((m_init = opt.m_init))
        construct(opt.get());
    return *this;
}

template <typename T>
optional<T>::operator bool() const {
    return m_init;
}

template <typename T>
T &optional<T>::operator *() {
    return get();
}

template <typename T>
const T &optional<T>::operator*() const {
    return get();
}

template <typename T>
void *optional<T>::storage() {
    return m_data;
}

template <typename T>
const void *optional<T>::storage() const {
    return m_data;
}

template <typename T>
T &optional<T>::get() {
    return *(detail::optional_cast<T, false> { storage() }).data;
}

template <typename T>
const T &optional<T>::get() const {
    return *(detail::optional_cast<T, true> { storage() }).data;
}

template <typename T>
void optional<T>::destruct() {
    if (m_init)
        get().~T();
    m_init = false;
}

template <typename T>
void optional<T>::construct(const T &data) {
    new (storage()) T(data);
}

// A little unique_ptr like file wrapper to achieve RAII. We can't use
// unique_ptr here because unique_ptr doesn't allow null default delters
struct file {
    file();
    file(FILE *fp);
    ~file();

    operator FILE*();
    FILE *get();

private:
    FILE *m_handle;
};


file::file()
    : m_handle(nullptr)
{
}

file::file(FILE *fp)
    : m_handle(fp)
{
}

file::~file() {
    if (m_handle)
        fclose(m_handle);
}

file::operator FILE*() {
    return m_handle;
}

FILE *file::get() {
    return m_handle;
}

// Wrapper around fopen for a u::string + path fixing
#ifdef _WIN32
static constexpr int kPathSep = '\\';
#else
static constexpr int kPathSep = '/';
#endif

static inline u::string fixPath(const u::string &path) {
#ifdef _WIN32
    u::string fix(path);
    const size_t size = fix.size();
    for (size_t i = 0; i < size; i++) {
        if (!strchr("/\\", fix[i]))
            continue;
        fix[i] = u::kPathSep;
    }
    return fix;
#endif
    return path;
}

u::file fopen(const u::string& infile, const char *type) {
    return ::fopen(fixPath(infile).c_str(), type);
}

// An imeplementation of getline for u::string on a u::file
u::optional<u::string> getline(u::file &fp) {
    u::string s;
    for (;;) {
        char buf[256];
        if (!fgets(buf, sizeof(buf), fp.get())) {
            if (feof(fp.get())) {
                if (s.empty())
                    return u::none;
                else
                    return u::move(s);
            }
            abort();
        }
        size_t n = strlen(buf);
        if (n && buf[n - 1] == '\n')
            --n;
        s.append(buf, n);
        if (n < sizeof(buf) - 1)
            return u::move(s);
    }
}

// A C99 compatible sscanf
namespace detail {
    int c99vsscanf(const char *s, const char *format, va_list ap) {
#if defined(_WIN32) || defined(_WIN64)
        u::string fmt = format;
#ifdef _WIN32
        replace_all(fmt, "%zu", "%u");
#else
        replace_all(fmt, "%zu", "%Iu");
#endif
#else
        const char *fmt = format;
#endif
        return vsscanf(s, &fmt[0], ap);
    }
}

inline int sscanf(const u::string &thing, const char *fmt, ...) {
    va_list va;
    va_start(va, fmt);
    int value = detail::c99vsscanf(thing.c_str(), fmt, va);
    va_end(va);
    return value;
}

// A way to tokenize a string
inline u::vector<u::string> split(const char *str, char ch = ' ') {
    u::vector<u::string> result;
    do {
        const char *begin = str;
        while (*str != ch && *str)
            str++;
        result.push_back(u::string(begin, str));
    } while (*str++);
    return result;
}

inline u::vector<u::string> split(const u::string &str, char ch = ' ') {
    return u::split(str.c_str(), ch);
}

}

// Maths
namespace m {

static const float kEpsilon = 0.00001f;

// Implementation of clamp
template <typename T>
inline T clamp(const T& current, const T &min, const T &max) {
    return (current > max) ? max : ((current < min) ? min : current);
}

// Fast absolute values
inline float abs(float v) {
    union {
        float f;
        int b;
    } data = { v };
    data.b &= 0x7FFFFFFF;
    return data.f;
}

// Implementation of a vec3
struct vec3 {
    union {
        struct {
            float x;
            float y;
            float z;
        };
        float m[3];
    };

    constexpr vec3();
    constexpr vec3(float nx, float ny, float nz);
    constexpr vec3(float a);

    float absSquared() const;
    float abs() const;

    void normalize();
    vec3 normalized() const;
    bool isNormalized() const;
    bool isNull() const;

    bool isNullEpsilon(const float epsilon = kEpsilon) const;
    bool equals(const vec3 &cmp, const float epsilon) const;

    void setLength(float scaleLength);
    void maxLength(float length);

    vec3 cross(const vec3 &v) const;

    vec3 &operator +=(const vec3 &vec);
    vec3 &operator -=(const vec3 &vec);
    vec3 &operator *=(float value);
    vec3 &operator /=(float value);
    vec3 operator -() const;
    float operator[](size_t index) const;
    float &operator[](size_t index);

    static const vec3 xAxis;
    static const vec3 yAxis;
    static const vec3 zAxis;
    static const vec3 origin;
};

const vec3 vec3::xAxis(1.0f, 0.0f, 0.0f);
const vec3 vec3::yAxis(0.0f, 1.0f, 0.0f);
const vec3 vec3::zAxis(0.0f, 0.0f, 1.0f);
const vec3 vec3::origin(0.0f, 0.0f, 0.0f);

inline constexpr vec3::vec3()
    : x(0.0f)
    , y(0.0f)
    , z(0.0f)
{
}

inline constexpr vec3::vec3(float nx, float ny, float nz)
    : x(nx)
    , y(ny)
    , z(nz)
{
}

inline constexpr vec3::vec3(float a)
    : x(a)
    , y(a)
    , z(a)
{
}

inline float vec3::absSquared() const {
    return x * x + y * y + z * z;
}

inline float vec3::abs() const {
    return sqrtf(x * x + y * y + z * z);
}

inline void vec3::normalize() {
    const float length = 1.0f / abs();
    x *= length;
    y *= length;
    z *= length;
}

inline vec3 vec3::normalized() const {
    const float scale = 1.0f / abs();
    return vec3(x * scale, y * scale, z * scale);
}

inline bool vec3::isNormalized() const {
    return m::abs(abs() - 1.0f) < kEpsilon;
}

inline bool vec3::isNull() const {
    return x == 0.0f && y == 0.0f && z == 0.0f;
}

inline bool vec3::isNullEpsilon(const float epsilon) const {
    return equals(origin, epsilon);
}

inline bool vec3::equals(const vec3 &cmp, const float epsilon) const {
    return (m::abs(x - cmp.x) < epsilon)
        && (m::abs(y - cmp.y) < epsilon)
        && (m::abs(z - cmp.z) < epsilon);
}

inline void vec3::setLength(float scaleLength) {
    const float length = scaleLength / abs();
    x *= length;
    y *= length;
    z *= length;
}

inline void vec3::maxLength(float length) {
    const float currentLength = abs();
    if (currentLength > length) {
        const float scale = length / currentLength;
        x *= scale;
        y *= scale;
        z *= scale;
    }
}

inline vec3 vec3::cross(const vec3 &v) const {
    return vec3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}

inline vec3 &vec3::operator +=(const vec3 &vec) {
    x += vec.x;
    y += vec.y;
    z += vec.z;
    return *this;
}

inline vec3 &vec3::operator -=(const vec3 &vec) {
    x -= vec.x;
    y -= vec.y;
    z -= vec.z;
    return *this;
}

inline vec3 &vec3::operator *=(float value) {
    x *= value;
    y *= value;
    z *= value;
    return *this;
}

inline vec3 &vec3::operator /=(float value) {
    const float inv = 1.0f / value;
    x *= inv;
    y *= inv;
    z *= inv;
    return *this;
}

inline vec3 vec3::operator -() const {
    return vec3(-x, -y, -z);
}

inline float vec3::operator[](size_t index) const {
    return m[index];
}

inline float &vec3::operator[](size_t index) {
    return m[index];
}

inline vec3 operator+(const vec3 &a, const vec3 &b) {
    return vec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline vec3 operator-(const vec3 &a, const vec3 &b) {
    return vec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline vec3 operator*(const vec3 &a, float value) {
    return vec3(a.x * value, a.y * value, a.z * value);
}

inline vec3 operator*(float value, const vec3 &a) {
    return vec3(a.x * value, a.y * value, a.z * value);
}

inline vec3 operator/(const vec3 &a, float value) {
    const float inv = 1.0f / value;
    return vec3(a.x * inv, a.y * inv, a.z * inv);
}

inline vec3 operator^(const vec3 &a, const vec3 &b) {
    return vec3((a.y * b.z - a.z * b.y),
                (a.z * b.x - a.x * b.z),
                (a.x * b.y - a.y * b.x));
}

inline float operator*(const vec3 &a, const vec3 &b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline bool operator==(const vec3 &a, const vec3 &b) {
    return (fabs(a.x - b.x) < kEpsilon)
        && (fabs(a.y - b.y) < kEpsilon)
        && (fabs(a.z - b.z) < kEpsilon);
}

inline bool operator!=(const vec3 &a, const vec3 &b) {
    return (m::abs(a.x - b.x) > kEpsilon)
        || (m::abs(a.y - b.y) > kEpsilon)
        || (m::abs(a.z - b.z) > kEpsilon);
}

inline vec3 clamp(const vec3 &current, const vec3 &min, const vec3 &max) {
    return { clamp(current.x, min.x, max.x),
             clamp(current.y, min.y, max.y),
             clamp(current.z, min.z, max.z) };
}

}

// Tangent stuff
static void calculateTangent(const u::vector<m::vec3> &vertices,
                             const u::vector<m::vec3> &coordinates,
                             size_t v0,
                             size_t v1,
                             size_t v2,
                             m::vec3 &tangent,
                             m::vec3 &bitangent)
{
    const m::vec3 &x = vertices[v0];
    const m::vec3 &y = vertices[v1];
    const m::vec3 &z = vertices[v2];
    const m::vec3 q1(y - x);
    const m::vec3 q2(z - x);
    const float s1 = coordinates[v1].x - coordinates[v0].x;
    const float s2 = coordinates[v2].x - coordinates[v0].x;
    const float t1 = coordinates[v1].y - coordinates[v0].y;
    const float t2 = coordinates[v2].y - coordinates[v0].y;
    const float det = s1*t2 - s2*t1;
    if (m::abs(det) <= m::kEpsilon) {
        // Unable to compute tangent + bitangent, default tangent along xAxis and
        // bitangent along yAxis.
        tangent = m::vec3::xAxis;
        bitangent = m::vec3::yAxis;
        return;
    }

    const float inv = 1.0f / det;
    tangent = m::vec3(inv * (t2 * q1.x - t1 * q2.x),
                      inv * (t2 * q1.y - t1 * q2.y),
                      inv * (t2 * q1.z - t1 * q2.z));
    bitangent = m::vec3(inv * (-s2 * q1.x + s1 * q2.x),
                        inv * (-s2 * q1.y + s1 * q2.y),
                        inv * (-s2 * q1.z + s1 * q2.z));
}

static void createTangents(const u::vector<m::vec3> &vertices,
                           const u::vector<m::vec3> &coordinates,
                           const u::vector<m::vec3> &normals,
                           const u::vector<size_t> &indices,
                           u::vector<m::vec3> &tangents_,
                           u::vector<float> &bitangents_)
{
    // Computing Tangent Space Basis Vectors for an Arbitrary Mesh (Lengyel’s Method)
    // Section 7.8 (or in Section 6.8 of the second edition).
    const size_t vertexCount = vertices.size();
    u::vector<m::vec3> tangents;
    u::vector<m::vec3> bitangents;

    tangents.resize(vertexCount);
    bitangents.resize(vertexCount);

    m::vec3 tangent;
    m::vec3 bitangent;

    for (size_t i = 0; i < indices.size(); i += 3) {
        const size_t x = indices[i+0];
        const size_t y = indices[i+1];
        const size_t z = indices[i+2];

        calculateTangent(vertices, coordinates, x, y, z, tangent, bitangent);

        tangents[x] += tangent;
        tangents[y] += tangent;
        tangents[z] += tangent;
        bitangents[x] += bitangent;
        bitangents[y] += bitangent;
        bitangents[z] += bitangent;
    }

    for (size_t i = 0; i < vertexCount; i++) {
        // Gram-Schmidt orthogonalize
        // http://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
        const m::vec3 &n = normals[i];
        m::vec3 t = tangents[i];
        const m::vec3 v = (t - n * (n * t));
        tangents_[i] = v.isNullEpsilon() ? v : v.normalized();

        if (!tangents_[i].isNormalized()) {
            // Couldn't calculate vertex tangent for vertex, so we fill it in along
            // the x axis.
            tangents_[i] = m::vec3::xAxis;
            t = tangents_[i];
        }

        // bitangents are only stored by handness in the W component (-1.0f or 1.0f).
        bitangents_[i] = (((n ^ t) * bitangents[i]) < 0.0f) ? -1.0f : 1.0f;
    }
}

// Linear-speed vertex cache optimizer
struct vertexCacheData {
    u::vector<size_t> indices;
    size_t cachePosition;
    float currentScore;
    size_t totalValence;
    size_t remainingValence;
    bool calculated;

    vertexCacheData();

    size_t findTriangle(size_t tri);
    void moveTriangle(size_t tri);
};

struct triangleCacheData {
    bool rendered;
    float currentScore;
    size_t vertices[3];
    bool calculated;

    triangleCacheData();
};

struct vertexCache {
    void addVertex(size_t vertex);
    void clear();
    size_t getCacheMissCount() const;
    size_t getCacheMissCount(const u::vector<size_t> &indices);
    size_t getCachedVertex(size_t index) const;
    vertexCache();
private:
    size_t findVertex(size_t vertex);
    void removeVertex(size_t stackIndex);
    size_t m_cache[40];
    size_t m_misses;
};

struct vertexCacheOptimizer {
    vertexCacheOptimizer();

    enum result {
        kSuccess,
        kErrorInvalidIndex,
        kErrorNoVertices
    };

    result optimize(u::vector<size_t> &indices);

    size_t getCacheMissCount() const;

private:
    u::vector<vertexCacheData> m_vertices;
    u::vector<triangleCacheData> m_triangles;
    u::vector<size_t> m_indices;
    u::vector<size_t> m_drawList;
    vertexCache m_vertexCache;
    size_t m_bestTriangle;

    float calcVertexScore(size_t vertex);
    size_t fullScoreRecalculation();
    result initialPass();
    result init(u::vector<size_t> &indices, size_t vertexCount);
    void addTriangle(size_t triangle);
    bool cleanFlags();
    void triangleScoreRecalculation(size_t triangle);
    size_t partialScoreRecalculation();
    bool process();
};

struct face {
    face();
    size_t vertex;
    size_t normal;
    size_t coordinate;
};

inline face::face()
    : vertex((size_t)-1)
    , normal((size_t)-1)
    , coordinate((size_t)-1)
{
}

inline bool operator==(const face &lhs, const face &rhs) {
    return lhs.vertex == rhs.vertex
        && lhs.normal == rhs.normal
        && lhs.coordinate == rhs.coordinate;
}

// Hash a face
namespace std {
template <>
struct hash<::face> {
    size_t operator()(const ::face &f) const {
        static constexpr size_t prime1 = 73856093u;
        static constexpr size_t prime2 = 19349663u;
        static constexpr size_t prime3 = 83492791u;
        return (f.vertex * prime1) ^ (f.normal * prime2) ^ (f.coordinate * prime3);
    }
};
}

///! vertexCacheData
size_t vertexCacheData::findTriangle(size_t triangle) {
    for (size_t i = 0; i < indices.size(); i++)
        if (indices[i] == triangle)
            return i;
    return (size_t)-1;
}

void vertexCacheData::moveTriangle(size_t triangle) {
    size_t index = findTriangle(triangle);
    assert(index != (size_t)-1);

    // Erase the index and add it to the end
    indices.erase(indices.begin() + index,
                  indices.begin() + index + 1);
    indices.push_back(triangle);
}

vertexCacheData::vertexCacheData()
    : cachePosition((size_t)-1)
    , currentScore(0.0f)
    , totalValence(0)
    , remainingValence(0)
    , calculated(false)
{
}

///!triangleCacheData
triangleCacheData::triangleCacheData()
    : rendered(false)
    , currentScore(0.0f)
    , calculated(false)
{
    vertices[0] = (size_t)-1;
    vertices[1] = (size_t)-1;
    vertices[2] = (size_t)-1;
}

///! vertexCache
size_t vertexCache::findVertex(size_t vertex) {
    for (size_t i = 0; i < 32; i++)
        if (m_cache[i] == vertex)
            return i;
    return (size_t)-1;
}

void vertexCache::removeVertex(size_t stackIndex) {
    for (size_t i = stackIndex; i < 38; i++)
        m_cache[i] = m_cache[i + 1];
}

void vertexCache::addVertex(size_t vertex) {
    size_t w = findVertex(vertex);
    // remove the vertex for later reinsertion at the top
    if (w != (size_t)-1)
        removeVertex(w);
    else // not found, cache miss!
        m_misses++;
    // shift all vertices down to make room for the new top vertex
    for (size_t i = 39; i != 0; i--)
        m_cache[i] = m_cache[i - 1];
    // add new vertex to cache
    m_cache[0] = vertex;
}

void vertexCache::clear() {
    for (size_t i = 0; i < 40; i++)
        m_cache[i] = (size_t)-1;
    m_misses = 0;
}

size_t vertexCache::getCacheMissCount() const {
    return m_misses;
}

size_t vertexCache::getCacheMissCount(const u::vector<size_t> &indices) {
    clear();
    for (auto &it : indices)
        addVertex(it);
    return m_misses;
}

size_t vertexCache::getCachedVertex(size_t index) const {
    return m_cache[index];
}

vertexCache::vertexCache() {
    clear();
}

///! vertexCacheOptimizer
static constexpr float kCacheDecayPower = 1.5f;
static constexpr float kLastTriScore = 0.75f;
static constexpr float kValenceBoostScale = 2.0f;
static constexpr float kValenceBoostPower = 0.5f;

vertexCacheOptimizer::vertexCacheOptimizer()
    : m_bestTriangle(0)
{
}

vertexCacheOptimizer::result vertexCacheOptimizer::optimize(u::vector<size_t> &indices) {
    size_t find = (size_t)-1;
    for (size_t i = 0; i < indices.size(); i++)
        if (find == (size_t)-1 || (find != (size_t)-1 && indices[i] > find))
            find = indices[i];
    if (find == (size_t)-1)
        return kErrorNoVertices;

    result begin = init(indices, find + 1);
    if (begin != kSuccess)
        return begin;

    // Process
    while (process())
        ;

    // Rewrite the indices
    for (size_t i = 0; i < m_drawList.size(); i++)
        for (size_t j = 0; j < 3; j++)
            indices[3 * i + j] = m_triangles[m_drawList[i]].vertices[j];

    return kSuccess;
}

float vertexCacheOptimizer::calcVertexScore(size_t vertex) {
    vertexCacheData *v = &m_vertices[vertex];
    if (v->remainingValence == (size_t)-1 || v->remainingValence == 0)
        return -1.0f; // No triangle needs this vertex

    float value = 0.0f;
    if (v->cachePosition == (size_t)-1) {
        // Vertex is not in FIFO cache.
    } else {
        if (v->cachePosition < 3) {
            // This vertex was used in the last triangle. It has fixed score
            // in whichever of the tree it's in.
            value = kLastTriScore;
        } else {
            // Points for being heigh in the cache
            const float scale = 1.0f / (32 - 3);
            value = 1.0f - (v->cachePosition - 3) * scale;
            value = powf(value, kCacheDecayPower);
        }
    }

    // Bonus points for having a low number of triangles.
    float valenceBoost = powf(float(v->remainingValence), -kValenceBoostPower);
    value += kValenceBoostScale * valenceBoost;
    return value;
}

size_t vertexCacheOptimizer::fullScoreRecalculation() {
    // Calculate score for all vertices
    for (size_t i = 0; i < m_vertices.size(); i++)
        m_vertices[i].currentScore = calcVertexScore(i);

    // Calculate scores for all active triangles
    float maxScore = 0.0f;
    size_t maxScoreTriangle = (size_t)-1;
    bool firstTime = true;

    for (size_t i = 0; i < m_triangles.size(); i++) {
        auto &it = m_triangles[i];
        if (it.rendered)
            continue;

        // Sum the score of all the triangle's vertices
        float sum = m_vertices[it.vertices[0]].currentScore +
                    m_vertices[it.vertices[1]].currentScore +
                    m_vertices[it.vertices[2]].currentScore;
        it.currentScore = sum;

        if (firstTime || sum > maxScore) {
            firstTime = false;
            maxScore = sum;
            maxScoreTriangle = i;
        }
    }

    return maxScoreTriangle;
}

vertexCacheOptimizer::result vertexCacheOptimizer::initialPass() {
    for (size_t i = 0; i < m_indices.size(); i++) {
        size_t index = m_indices[i];
        if (index == (size_t)-1 || index >= m_vertices.size())
            return kErrorInvalidIndex;
        m_vertices[index].totalValence++;
        m_vertices[index].remainingValence++;
        m_vertices[index].indices.push_back(i / 3);
    }
    m_bestTriangle = fullScoreRecalculation();
    return kSuccess;
}

vertexCacheOptimizer::result vertexCacheOptimizer::init(u::vector<size_t> &indices, size_t maxVertex) {
    const size_t triangleCount = indices.size() / 3;

    // Reset draw list
    m_drawList.clear();
    m_drawList.reserve(maxVertex);

    // Reset and initialize vertices and triangles
    m_vertices.clear();
    m_vertices.reserve(maxVertex);
    for (size_t i = 0; i < maxVertex; i++)
        m_vertices.push_back(vertexCacheData());

    m_triangles.clear();
    m_triangles.reserve(triangleCount);
    for (size_t i = 0; i < triangleCount; i++) {
        triangleCacheData data;
        for (size_t j = 0; j < 3; j++)
            data.vertices[j] = indices[i * 3 + j];
        m_triangles.push_back(data);
    }

    // Copy the indices
    m_indices.clear();
    m_indices.reserve(indices.size());
    for (auto &it : indices)
        m_indices.push_back(it);

    // Run the initial pass
    m_vertexCache.clear();
    m_bestTriangle = (size_t)-1;

    return initialPass();
}

void vertexCacheOptimizer::addTriangle(size_t triangle) {
    // reset all cache positions
    for (size_t i = 0; i < 32; i++) {
        size_t find = m_vertexCache.getCachedVertex(i);
        if (find == (size_t)-1)
            continue;
        m_vertices[find].cachePosition = (size_t)-1;
    }

    triangleCacheData *t = &m_triangles[triangle];
    if (t->rendered)
        return;

    for (size_t i = 0; i < 3; i++) {
        // Add all the triangle's vertices to the cache
        m_vertexCache.addVertex(t->vertices[i]);
        vertexCacheData *v = &m_vertices[t->vertices[i]];

        // Decrease the remaining valence.
        v->remainingValence--;

        // Move the added triangle to the end of the vertex's triangle index
        // list such that the first `remainingValence' triangles in the index
        // list are only the active ones.
        v->moveTriangle(triangle);
    }

    // It's been rendered, mark it
    m_drawList.push_back(triangle);
    t->rendered = true;

    // Update all the vertex cache positions
    for (size_t i = 0; i < 32; i++) {
        size_t index = m_vertexCache.getCachedVertex(i);
        if (index == (size_t)-1)
            continue;
        m_vertices[index].cachePosition = i;
    }
}

// Avoid duplicate calculations during processing. Triangles and vertices have
// a `calculated' flag which must be reset at the beginning of the process for
// all active triangles that have one or more of their vertices currently in
// cache as well all their other vertices.
//
// If there aren't any active triangles in the cache this function returns
// false and a full recalculation of the tree is performed.
bool vertexCacheOptimizer::cleanFlags() {
    bool found = false;
    for (size_t i = 0; i < 32; i++) {
        size_t find = m_vertexCache.getCachedVertex(i);
        if (find == (size_t)-1)
            continue;

        vertexCacheData *v = &m_vertices[find];
        for (size_t j = 0; j < v->remainingValence; j++) {
            triangleCacheData *t = &m_triangles[v->indices[j]];
            found = true;
            // Clear flags
            t->calculated = false;
            for (size_t k = 0; k < 3; k++)
                m_vertices[t->vertices[k]].calculated = false;
        }
    }
    return found;
}

void vertexCacheOptimizer::triangleScoreRecalculation(size_t triangle) {
    triangleCacheData *t = &m_triangles[triangle];

    // Calculate vertex scores
    float sum = 0.0f;
    for (size_t i = 0; i < 3; i++) {
        vertexCacheData *v = &m_vertices[t->vertices[i]];
        float score = v->calculated ? v->currentScore : calcVertexScore(t->vertices[i]);
        v->currentScore = score;
        v->calculated = true;
        sum += score;
    }

    t->currentScore = sum;
    t->calculated = true;
}

size_t vertexCacheOptimizer::partialScoreRecalculation() {
    // Iterate through all the vertices of the cache
    bool firstTime = true;
    float maxScore = 0.0f;
    size_t maxScoreTriangle = (size_t)-1;
    for (size_t i = 0; i < 32; i++) {
        size_t find = m_vertexCache.getCachedVertex(i);
        if (find == (size_t)-1)
            continue;

        vertexCacheData *v = &m_vertices[find];

        // Iterate through all the active triangles of this vertex
        for (size_t j = 0; j < v->remainingValence; j++) {
            size_t triangle = v->indices[j];
            triangleCacheData *t = &m_triangles[triangle];

            // Calculate triangle score if it isn't already calculated
            if (!t->calculated)
                triangleScoreRecalculation(triangle);

            float score = t->currentScore;
            // Found a triangle to process
            if (firstTime || score > maxScore) {
                firstTime = false;
                maxScore = score;
                maxScoreTriangle = triangle;
            }
        }
    }
    return maxScoreTriangle;
}

inline bool vertexCacheOptimizer::process() {
    if (m_drawList.size() == m_triangles.size())
        return false;

    // Add the selected triangle to the draw list
    addTriangle(m_bestTriangle);

    // Recalculate the vertex and triangle scores and select the best triangle
    // for the next iteration.
    m_bestTriangle = cleanFlags() ? partialScoreRecalculation() : fullScoreRecalculation();

    return true;
}

size_t vertexCacheOptimizer::getCacheMissCount() const {
    return m_vertexCache.getCacheMissCount();
}

// OBJ loading
struct obj {
    bool load(const u::string &file, int &before, int &after);
    u::vector<size_t> m_indices;
    u::vector<m::vec3> m_positions;
    u::vector<m::vec3> m_normals;
    u::vector<m::vec3> m_coordinates;
    u::vector<m::vec3> m_tangents;
    u::vector<float> m_bitangents; // Sign only
};

bool obj::load(const u::string &file, int &before, int &after) {
    u::file fp = u::fopen(file, "r");
    if (!fp)
        return false;

    // Processed vertices, normals and coordinates from the OBJ file
    u::vector<m::vec3> vertices;
    u::vector<m::vec3> normals;
    u::vector<m::vec3> coordinates;
    u::vector<m::vec3> tangents;
    u::vector<float> bitangents;

    // Unique vertices are stored in a map keyed by face.
    u::map<face, size_t> uniques;

    size_t count = 0;
    size_t group = 0;
    while (auto get = u::getline(fp)) {
        auto &line = *get;
        // Skip whitespace
        while (line.size() && strchr(" \t", line[0]))
            line.erase(line.begin());
        // Skip comments
        if (strchr("#$", line[0]))
            continue;
        // Skip empty lines
        if (line.empty())
            continue;

        // Process the individual lines
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        if (u::sscanf(line, "v %f %f %f", &x, &y, &z) == 3) {
            // v float float float
            vertices.push_back({x, y, z * -1.0f});
        } else if (u::sscanf(line, "vn %f %f %f", &x, &y, &z) == 3) {
            // vn float float float
            normals.push_back({x * -1.0f, y * -1.0f, z});
        } else if (u::sscanf(line, "vt %f %f", &x, &y) == 2) {
            // vt float float
            coordinates.push_back({x, 1.0f - y, 0.0f});
        } else if (line[0] == 'g') {
            group++;
        } else if (line[0] == 'f' && group == 0) { // Only process the first group faces
            u::vector<size_t> v;
            u::vector<size_t> n;
            u::vector<size_t> t;

            // Note: 1 to skip "f"
            auto contents = u::split(line);
            for (size_t i = 1; i < contents.size(); i++) {
                int vi = 0;
                int ni = 0;
                int ti = 0;
                if (u::sscanf(contents[i], "%i/%i/%i", &vi, &ti, &ni) == 3) {
                    v.push_back(vi < 0 ? v.size() + vi : vi - 1);
                    t.push_back(ti < 0 ? t.size() + ti : ti - 1);
                    n.push_back(ni < 0 ? n.size() + ni : ni - 1);
                } else if (u::sscanf(contents[i], "%i//%i", &vi, &ni) == 2) {
                    v.push_back(vi < 0 ? v.size() + vi : vi - 1);
                    n.push_back(ni < 0 ? n.size() + ni : ni - 1);
                } else if (u::sscanf(contents[i], "%i/%i", &vi, &ti) == 2) {
                    v.push_back(vi < 0 ? v.size() + vi : vi - 1);
                    t.push_back(ti < 0 ? t.size() + ti : ti - 1);
                } else if (u::sscanf(contents[i], "%i", &vi) == 1) {
                    v.push_back(vi < 0 ? v.size() + vi : vi - 1);
                }
            }

            // Triangulate the mesh
            for (size_t i = 1; i < v.size() - 1; ++i) {
                auto index = m_indices.size();
                m_indices.resize(index + 3);
                auto triangulate = [&v, &n, &t, &uniques, &count](size_t index, size_t &out) {
                    face triangle;
                    triangle.vertex = v[index];
                    if (n.size()) triangle.normal = n[index];
                    if (t.size()) triangle.coordinate = t[index];
                    // Only insert in the map if it doesn't exist
                    if (uniques.find(triangle) == uniques.end())
                        uniques[triangle] = count++;
                    out = uniques[triangle];
                };
                triangulate(0,     m_indices[index + 0]);
                triangulate(i + 0, m_indices[index + 1]);
                triangulate(i + 1, m_indices[index + 2]);
            }
        }
    }

    // Construct the model, indices are already generated
    m_positions.resize(count);
    m_normals.resize(count);
    m_coordinates.resize(count);
    for (auto &it : uniques) {
        const auto &first = it.first;
        const auto &second = it.second;
        m_positions[second] = vertices[first.vertex];
        if (normals.size())
            m_normals[second] = normals[first.normal];
        if (coordinates.size())
            m_coordinates[second] = coordinates[first.coordinate];
    }

    // Optimize the indices
    vertexCache cache;
    vertexCacheOptimizer vco;
    vco.optimize(m_indices);

    // Change winding order
    for (size_t i = 0; i < m_indices.size(); i += 3)
        u::swap(m_indices[i], m_indices[i + 2]);

    // Calculate tangents
    m_tangents.resize(count);
    m_bitangents.resize(count);
    createTangents(m_positions, m_coordinates, m_normals, m_indices, m_tangents, m_bitangents);

    before = cache.getCacheMissCount(m_indices);
    after = vco.getCacheMissCount();

    return true;
}

#include "smbf.h"
int main(int argc, char **argv) {
    --argc;
    ++argv;
    bool tangents = false;
    bool bitangents = false;
    const char *file = nullptr;

    for (int i = 0; i < argc; i++) {
        if (!strcmp(argv[i], "-t"))
            tangents = true;
        else if (!strcmp(argv[i], "-b"))
            bitangents = true;
        else
            file = argv[i];
    }

    if (!file) {
        fprintf(stderr, "expected OBJ file name\n");
        return 1;
    }

    int before = 0;
    int after = 0;
    obj o;
    if (!o.load(file, before, after)) {
        fprintf(stderr, "failed to load OBJ model `%s'\n", file);
        return 1;
    }

    int format = SMBF_FORMAT_P;
    if (o.m_normals.size())
        format = SMBF_FORMAT_PN;
    if (o.m_coordinates.size())
        format = SMBF_FORMAT_PNC;
    if (o.m_tangents.size() && tangents)
        format = SMBF_FORMAT_PNCT;
    if (o.m_bitangents.size() && bitangents)
        format = SMBF_FORMAT_PNCTB;


    const char *formats[] = {
        "P", "PN", "PNC", "PNCT", "PNCTB"
    };

    header h;
    h.magic[0] = 'S';
    h.magic[1] = 'M';
    h.magic[2] = 'B';
    h.magic[3] = 'F';
    h.version = SMBF_VERSION;
    h.format = format;
    h.count = o.m_positions.size();
    h.indices = o.m_indices.size();

    // Interleave the data
    union data {
        smbfP asP;
        smbfPN asPN;
        smbfPNC asPNC;
        smbfPNCT asPNCT;
        smbfPNCTB asPNCTB;
    };
    u::vector<data> inter;
    inter.resize(h.count);
    for (size_t i = 0; i < h.count; i++) {
        data &x = inter[i];
        smbfPNCTB &s = x.asPNCTB;
        s.px = o.m_positions[i].x;
        s.py = o.m_positions[i].y;
        s.pz = o.m_positions[i].z;
        if (o.m_normals.size()) {
            s.nx = o.m_normals[i].x;
            s.ny = o.m_normals[i].y;
            s.nz = o.m_normals[i].z;
        }
        if (o.m_coordinates.size()) {
            s.tu = o.m_coordinates[i].x;
            s.tv = o.m_coordinates[i].y;
        }
        s.tx = o.m_tangents[i].x;
        s.ty = o.m_tangents[i].y;
        s.tz = o.m_tangents[i].z;
        s.b = o.m_bitangents[i];
    }

    u::string name(file, strrchr(file, '.'));
    name += ".smbf";
    u::file write = u::fopen(name, "w");
    if (!write) {
        fprintf(stderr, "failed to open `%s' for writing\n", name.c_str());
        return 1;
    }

    fwrite(&h, sizeof(header), 1, write);
    for (auto &it : inter) {
        switch (format) {
        case SMBF_FORMAT_P:
            fwrite(&it.asP, sizeof(smbfP), 1, write);
            break;
        case SMBF_FORMAT_PN:
            fwrite(&it.asPN, sizeof(smbfPN), 1, write);
            break;
        case SMBF_FORMAT_PNC:
            fwrite(&it.asPNC, sizeof(smbfPNC), 1, write);
            break;
        case SMBF_FORMAT_PNCT:
            fwrite(&it.asPNCT, sizeof(smbfPNCT), 1, write);
            break;
        case SMBF_FORMAT_PNCTB:
            fwrite(&it.asPNCTB, sizeof(smbfPNCTB), 1, write);
            break;
        }
    }

    for (auto &it : o.m_indices) {
        uint32_t convert = it;
        fwrite(&convert, sizeof(uint32_t), 1, write);
    }

    printf("Wrote `%s':\n", name.c_str());
    printf(" Indices:\n");
    printf("  Count: %i\n", int(h.indices));
    printf("  Cache misses:\n");
    printf("   Before: %i\n", before);
    printf("   After:  %i\n", after);
    printf(" Vertices:\n");
    printf("  Count:  %i\n", int(h.count));
    printf("  Format: %s\n", formats[format]);
    return 0;
}

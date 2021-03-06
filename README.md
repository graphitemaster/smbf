# SMBF - Static model binary format

Static model binary format is just as you expect. A static model format
in binary format designed for renderers.

Unlike most model formats. SMBF puts an emphasis on being straight forward and
slim. There is no bounding box information, no vertex-edge information, or
anything of that nature. SMBF files contain just the data you need for rendering.
This data includes any of the following things:

* Position
* Normals (optional)
* Texture coordinates (optional)
* Tangents (optional)
* Bitangents (optional)

The data provided by SMBF is already in interleaved vertex format because any
other format is wrong.

The format is so straight forward you don't need any complex API to use, since
rolling a loader for it takes only a couple of lines. We provide a structure of
vertex definitions for you, but you don't have to use them.

You may be questioning why something so obvious needs to exist at all when it
should already exist. The reality is lots of people roll their own custom formats
specifically like this but never make them publicly available for others to use
because they're considered too trivial. This constant proliferation is
ridiculous, lets at least have a format for others to use too guys.

# Vertex formats

The following formats exist:

* P
* PN
* PNC
* PNCT
* PNCTB

Symbolically the format refers to these formats with the following enum:
```
enum {
    SMBF_FORMAT_P,
    SMBF_FORMAT_PN,
    SMBF_FORMAT_PNC,
    SMBF_FORMAT_PNCT,
    SMBF_FORMAT_PNCTB
};
```

## Legend

* The P is short for position
* The N is short for normal
* The C is short for coordinate (as in texture coordinate.)
* The T is short for tangent
* The B is short for bitangent

## Storage of data

* P is 3-component float vector.
* N is 3-component float vector.
* C is 2-component float vector.
* T is 3-component float vector.
* B is 1-component float storing sign of `W`

The reason we only store sign for bitangents is because the cost of bandwidth
far exceeds the cost of a single cross produt. So just reconstruct it like
so.

```
    normal0 = (gWorld * vec4(normal, 0.0f)).xyz;
    tangent0 = (gWorld * vec4(tangent, 0.0f)).xyz;
    bitangent0 = w * cross(normal0, tangent0);
```

# File format

The file format contains the following header

```
struct header {
    uint8_t magic[4]; // Always "SMBF"
    uint8_t format; // The vertex format (one of SMFB_FORMAT_*)
    uint16_t version; // File format version
    uint32_t count; // How many "vertices"
    uint32_t indices; // How many "indices"
};
```

After the header there is `count` vertices of type `format`.
After the vertices there is `indices` indices of type `uint32_t`.

Everything is written as Little Endian.

# Bounding Boxes
SMBF does not concern itself with storing information that is trivial to calculate
at runtime and often is faster than the wasted storage space. Since SMBF is for
static models only it's trivial to construct a bounding box; here's an example:
```
// calculate the amount of floats for the format type
size_t size = 0;
if (type == SMBF_FORMAT_P)     size = sizeof(struct smbfP)     / sizeof(float);
if (type == SMBF_FORMAT_PN)    size = sizeof(struct smbfPN)    / sizeof(float);
if (type == SMBF_FORMAT_PNC)   size = sizeof(struct smbfPNC)   / sizeof(float);
if (type == SMBF_FORMAT_PNCT)  size = sizeof(struct smbfPNCT)  / sizeof(float);
if (type == SMBF_FORMAT_PNCTB) size = sizeof(struct smbfPNCTB) / sizeof(float);

struct header* h = stream; // header
float *v = (float *)h + 1; // vertex data begins after the header

vec3 min(v[0], v[1], v[2]);
vec3 max(v[0], v[1], v[2]);
for (uint32_t i = 0; i < h->count; i++)
{
    for (size_t j = 0; j < 3; j++)
    {
        if (v[i][j] < min[j]) min[j] = v[i][j];
        if (v[i][j] > max[j]) max[j] = v[i][j];
    }
    // move forward in the vertex stream
    v += size;
}
```

# Toolkit

Included is a toolkit for converting OBJ models to SMBF models. To use the
tool just invoke it with the OBJ model as the first argument and it will spit
out a model of the same name with a .smbf extension. You may also enable generation
of tangents with the `-t` option and bitangents with `-b` option.

All models processed with the toolkit will have their indices optimized using
linear-speed-vertex-cache optimization to reduce potential cache misses when
used for rendering.

Toolkit does not produce valid files on Big Endian machines yet.

# License

The specification and the contents of `smbf.h` are placed in the public domain.
The toolkit code in `smbf.cpp` is released under the MIT license.

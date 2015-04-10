#ifndef SMBF_HDR
#define SMBF_HDR
#include <stdint.h>

#define SMBF_VERSION_MAJOR 1
#define SMBF_VERSION_MINOR 0

#define SMBF_VERSION (((SMBF_VERSION_MAJOR) << 8) | (SMBF_VERSION_MINOR))

struct header {
    uint8_t magic[4]; // Always "SMBF"
    uint8_t format; // The vertex format (one of SMFB_FORMAT_*)
    uint16_t version; // File format version
    uint32_t count; // How many "vertices"
    uint32_t indices; // How many "indices"
};

enum {
    SMBF_FORMAT_P,
    SMBF_FORMAT_PN,
    SMBF_FORMAT_PNC,
    SMBF_FORMAT_PNCT,
    SMBF_FORMAT_PNCTB
};

#pragma pack(push, 1)
struct smbfP {
    float px, py, pz;
};

struct smbfPN {
    float px, py, pz;
    float nx, ny, nz;
};

struct smbfPNC {
    float px, py, pz;
    float nx, ny, nz;
    float tu, tv;
};

struct smbfPNCT {
    float px, py, pz;
    float nx, ny, nz;
    float tu, tv;
    float tx, ty, tz;
};

struct smbfPNCTB {
    float px, py, pz;
    float nx, ny, nz;
    float tu, tv;
    float tx, ty, tz;
    float b;
};
#pragma pack(pop)

#endif

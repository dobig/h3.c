#include "h3_trace.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *trace_dir(void) {
    const char *dir = getenv("H3_TRACE_DIR");
    return dir && *dir ? dir : NULL;
}

int h3_trace_enabled(void) {
    return trace_dir() != NULL;
}

int h3_trace_wants(const char *name) {
    if (!trace_dir() || !name) return 0;
    const char *filter = getenv("H3_TRACE_FILTER");
    if (!filter || !*filter) return 1;
    while (*filter) {
        const char *comma = strchr(filter, ',');
        size_t length = comma ? (size_t)(comma - filter) : strlen(filter);
        if (length && !strncmp(name, filter, length)) return 1;
        if (!comma) break;
        filter = comma + 1;
    }
    return 0;
}

static int valid_name(const char *name) {
    if (!*name) return 0;
    for (const char *at = name; *at; at++) {
        char c = *at;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
            return 0;
    }
    return name[0] != '.';
}

static int write_tensor(const char *name, const char *dtype, size_t item,
                        const void *values, int ndim,
                        const uint64_t *shape) {
    if (!h3_trace_wants(name)) return 1;
    if (!valid_name(name) || ndim < 1 || ndim > 8 || !shape || !values) {
        fprintf(stderr, "h3: trace: rejected tensor \"%s\"\n", name);
        return 0;
    }
    uint64_t elements = 1;
    for (int axis = 0; axis < ndim; axis++) {
        if (shape[axis] && elements > UINT64_MAX / shape[axis]) {
            fprintf(stderr, "h3: trace: %s shape overflows\n", name);
            return 0;
        }
        elements *= shape[axis];
    }
    if (elements > UINT64_MAX / item) {
        fprintf(stderr, "h3: trace: %s shape overflows\n", name);
        return 0;
    }
    uint64_t bytes = elements * item;

    char header[512];
    int used = snprintf(header, sizeof(header),
                        "{\"value\":{\"dtype\":\"%s\",\"shape\":[", dtype);
    for (int axis = 0; axis < ndim && used > 0 &&
                       (size_t)used < sizeof(header); axis++)
        used += snprintf(header + used, sizeof(header) - (size_t)used,
                         "%s%llu", axis ? "," : "",
                         (unsigned long long)shape[axis]);
    if (used > 0 && (size_t)used < sizeof(header))
        used += snprintf(header + used, sizeof(header) - (size_t)used,
                         "],\"data_offsets\":[0,%llu]}}",
                         (unsigned long long)bytes);
    if (used <= 0 || (size_t)used >= sizeof(header) - 8) {
        fprintf(stderr, "h3: trace: %s header too long\n", name);
        return 0;
    }
    /* Pad with spaces so the payload starts 8-byte aligned. */
    while (used % 8) header[used++] = ' ';

    const char *dir = trace_dir();
    (void)mkdir(dir, 0755);
    size_t path_size = strlen(dir) + strlen(name) + 32;
    char *path = malloc(path_size);
    char *partial = malloc(path_size);
    if (!path || !partial) {
        free(path);
        free(partial);
        fprintf(stderr, "h3: trace: out of memory for %s\n", name);
        return 0;
    }
    snprintf(path, path_size, "%s/%s.safetensors", dir, name);
    snprintf(partial, path_size, "%s/%s.safetensors.partial", dir, name);

    int ok = 0;
    FILE *file = fopen(partial, "wb");
    if (file) {
        uint64_t header_size = (uint64_t)used;
        unsigned char size_le[8];
        for (int index = 0; index < 8; index++)
            size_le[index] = (unsigned char)(header_size >> (8 * index));
        ok = fwrite(size_le, 1, 8, file) == 8 &&
             fwrite(header, 1, (size_t)used, file) == (size_t)used &&
             (bytes == 0 ||
              fwrite(values, 1, (size_t)bytes, file) == (size_t)bytes);
        ok = fclose(file) == 0 && ok;
        ok = ok && rename(partial, path) == 0;
        if (!ok) (void)remove(partial);
    }
    if (!ok)
        fprintf(stderr, "h3: trace: cannot write %s: %s\n", path,
                strerror(errno));
    free(path);
    free(partial);
    return ok;
}

int h3_trace_f32(const char *name, const float *values,
                 int ndim, const uint64_t *shape) {
    return write_tensor(name, "F32", sizeof(float), values, ndim, shape);
}

int h3_trace_bf16(const char *name, const uint16_t *values,
                  int ndim, const uint64_t *shape) {
    return write_tensor(name, "BF16", sizeof(uint16_t), values, ndim, shape);
}

int h3_trace_u8(const char *name, const uint8_t *values,
                int ndim, const uint64_t *shape) {
    return write_tensor(name, "U8", sizeof(uint8_t), values, ndim, shape);
}

int h3_trace_u32(const char *name, const uint32_t *values,
                 int ndim, const uint64_t *shape) {
    return write_tensor(name, "U32", sizeof(uint32_t), values, ndim, shape);
}

int h3_trace_gpu(const char *name, const h3_gpu_tensor *tensor,
                 uint64_t rows, uint64_t width) {
    if (!h3_trace_wants(name)) return 1;
    if (!tensor || (width && rows > SIZE_MAX / width)) return 0;
    size_t elements = (size_t)(rows * width);
    uint64_t shape[] = {rows, width};
    int ok = 0;
    switch (h3_gpu_tensor_dtype(tensor)) {
    case H3_GPU_BF16: {
        uint16_t *values = malloc(elements * sizeof(*values) + 1);
        ok = values && h3_gpu_tensor_read_bf16(tensor, values, elements) &&
             h3_trace_bf16(name, values, 2, shape);
        free(values);
        break;
    }
    case H3_GPU_F32: {
        float *values = malloc(elements * sizeof(*values) + 1);
        ok = values && h3_gpu_tensor_read_f32(tensor, values, elements) &&
             h3_trace_f32(name, values, 2, shape);
        free(values);
        break;
    }
    default:
        break;
    }
    if (!ok) fprintf(stderr, "h3: trace: cannot read back %s\n", name);
    return ok;
}

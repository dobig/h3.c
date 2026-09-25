#ifndef H3_TRACE_H
#define H3_TRACE_H

#include "h3_gpu.h"

#include <stddef.h>
#include <stdint.h>

/* Parity tracing. With H3_TRACE_DIR set, named intermediate tensors are
 * written as one safetensors file each, "<dir>/<name>.safetensors", holding a
 * single tensor called "value" with the exact bytes the pipeline produced.
 * Two runs (C and Rust, or before and after an optimization) are compared
 * name by name with h3-parity. Tracing adds GPU synchronization points, so
 * it is for parity runs only; with the variable unset every call is a
 * cheap no-op and the pipeline is unchanged.
 *
 * H3_TRACE_FILTER optionally limits tracing to names starting with one of its
 * comma-separated prefixes, e.g. "dit.step00.,latent." to keep a long run's
 * trace small. */

int h3_trace_enabled(void);
int h3_trace_wants(const char *name);

/* Shapes are outermost first; ndim is at most 8. Return 1 on success or when
 * tracing is off, 0 if the file could not be written (reported on stderr;
 * callers treat tracing failures as non-fatal). */
int h3_trace_f32(const char *name, const float *values,
                 int ndim, const uint64_t *shape);
int h3_trace_bf16(const char *name, const uint16_t *values,
                  int ndim, const uint64_t *shape);
int h3_trace_u8(const char *name, const uint8_t *values,
                int ndim, const uint64_t *shape);
int h3_trace_u32(const char *name, const uint32_t *values,
                 int ndim, const uint64_t *shape);

/* Read back the first rows*width elements of a GPU tensor and trace them as
 * [rows,width]. The caller must have submitted the work that produced it. */
int h3_trace_gpu(const char *name, const h3_gpu_tensor *tensor,
                 uint64_t rows, uint64_t width);

#endif

// Actual GGML CPU buffers and bridge callbacks; no checkpoint or inference.
#include "../native/llama-bridge/bridge.cpp"
#include "ggml-alloc.h"
#include <cassert>
#include <iostream>

static void exercise(ggml_backend_t backend, int prefix) {
  Engine e;
  e.layers = 2;
  e.inter = 4;
  e.hidden = 3;
  e.total = 66;
  e.visits.assign(2, 0);
  ScopedPush edits[] = {{0, 1, 0, 3.f, prefix + 63},
                        {0, 2, 1, 0.f, prefix + 64},
                        {1, 3, 2, -7.f, prefix + 65}};
  Selected selected[] = {{0, prefix + 63, 1}, {0, prefix + 64, 2}, {1, prefix + 65, 3}};
  float values[3] = {};
  Scoped s;
  s.position_offset = prefix;
  s.edits = edits;
  s.n_edits = 3;
  s.selected = selected;
  s.n_selected = 3;
  s.values = values;
  s.edit_visits.assign(3, 0);
  s.capture_visits.assign(3, 0);
  e.scoped = &s;
  for (int off : {0, 64}) {
    e.offset = off;
    int count = std::min(64, e.total - off);
    for (int layer = 0; layer < 2; ++layer) {
      auto *ctx = ggml_init({16384, nullptr, true});
      auto *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, count);
      ggml_set_name(t, ("ffn_swiglu-" + std::to_string(layer)).c_str());
      auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
      assert(buffer);
      std::vector<float> before(4 * count), actual(4 * count);
      for (int j = 0; j < count; ++j)
        for (int n = 0; n < 4; ++n)
          before[4*j+n] = float(100*layer + 4*(off+j) + n);
      ggml_backend_tensor_set(t, before.data(), 0, before.size()*sizeof(float));
      assert(callback(t, true, &e) && callback(t, false, &e));
      ggml_backend_tensor_get(t, actual.data(), 0, actual.size()*sizeof(float));
      for (int j = 0; j < count; ++j)
        for (int n = 0; n < 4; ++n) {
          float expected = before[4*j+n];
          if (layer == 0 && off+j == 63 && n == 1) expected += 3;
          if (layer == 0 && off+j == 64 && n == 2) expected = 0;
          if (layer == 1 && off+j == 65 && n == 3) expected = -7;
          assert(actual[4*j+n] == expected);
        }
      // A repeated selected callback must fail, never silently overwrite evidence.
      if (layer == 1 && off == 64) {
        assert(!callback(t, false, &e));
        assert(e.failure.find("Duplicate") != std::string::npos);
      }
      ggml_backend_buffer_free(buffer);
      ggml_free(ctx);
    }
  }
  assert(values[0] == 253 && values[1] == 258 && values[2] == 363);
  // A later ordinary request must not retain any scoped edit or capture pointer.
  assert(syn_clear(&e) == 0 && e.scoped == nullptr);
  auto *ctx = ggml_init({4096, nullptr, true});
  auto *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 1);
  ggml_set_name(t, "ffn_swiglu-0");
  assert(!callback(t, true, &e));
  ggml_free(ctx);
  float logits[1];
  assert(syn_eval_scoped(&e, nullptr, 1, true, 0, nullptr, 0,
                         edits, 3, selected, 3, values, logits) < 0);
  assert(e.scoped == nullptr);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  ggml_backend_load_all_from_path(argv[1]);
  auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  assert(device);
  auto backend = ggml_backend_dev_init(device, nullptr);
  assert(backend);
  exercise(backend, 0);
  exercise(backend, 100); // cached prefix uses absolute positions
  ggml_backend_free(backend);
  std::cout << "Native scoped capture/edit checks passed (synthetic CPU tensors).\n";
}

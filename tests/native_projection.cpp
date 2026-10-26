// Real callback and CPU tensor buffers; no model, context, download or inference.
#include "../native/llama-bridge/bridge.cpp"
#include "ggml-alloc.h"
#include <cassert>
#include <iostream>
#include <limits>

static void exercise(ggml_backend_t backend, float strength) {
  Engine e;
  e.layers = 2;
  e.hidden = 3;
  e.inter = 4;
  e.total = 66;
  e.resid_visits.assign(2, 0);
  float u[] = {1, 0, 0}, captured[3] = {};
  double stats[15] = {};
  Research r;
  r.layer = 1;
  r.token = 65;
  r.direction = u;
  r.strength = strength;
  r.vector = captured;
  r.stats = stats;
  e.research = &r;
  synapse::validate_direction(u, 3, strength);
  for (int offset : {0, 64}) {
    e.offset = offset;
    int count = std::min(64, e.total - offset);
    auto *ctx = ggml_init({16384, nullptr, true});
    std::vector<ggml_tensor *> tensors;
    for (auto name : {"model.input_embed", "linear_attn_out-0", "ffn_out-0",
                      "attn_output-1", "ffn_out-1", "l_out-1", "result_output"}) {
      auto *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, count);
      ggml_set_name(t, name);
      tensors.push_back(t);
    }
    auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    assert(buffer);
    for (size_t i = 0; i < tensors.size(); ++i) {
      std::vector<float> values(3 * count);
      for (int j = 0; j < count; ++j) {
        values[3*j] = float(offset + j + 1);
        values[3*j+1] = 2;
        values[3*j+2] = -3;
      }
      auto *t = tensors[i];
      ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
      bool wanted = callback(t, true, &e);
      assert(wanted == (i < 6));
      if (wanted)
        assert(callback(t, false, &e));
      std::vector<float> actual(values.size());
      ggml_backend_tensor_get(t, actual.data(), 0, actual.size() * sizeof(float));
      for (int j = 0; j < count; ++j) {
        const float expected = i < 5 ? values[3*j] * (1-strength) : values[3*j];
        assert(std::abs(actual[3*j] - expected) < 1e-6);
        assert(actual[3*j+1] == 2 && actual[3*j+2] == -3);
      }
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
  }
  assert(r.captured == 1 && captured[0] == 66 && captured[1] == 2 && captured[2] == -3);
  const double squares = 66. * 67. * 133. / 6.;
  for (int i = 0; i < 5; ++i) {
    assert(stats[3*i] == 66);
    assert(std::abs(stats[3*i+1] - squares * strength * strength) < 1e-6);
    assert(std::abs(stats[3*i+2] - (squares + 66 * 13)) < 1e-6);
  }
  // Without a direction, no write callbacks execute. Capturing a residual is
  // still allowed, and the output readout is never selected.
  r.direction = nullptr;
  auto *ctx = ggml_init({4096, nullptr, true});
  auto *t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 3, 1);
  ggml_set_name(t, "ffn_out-1");
  assert(!callback(t, true, &e));
  ggml_set_name(t, "l_out-1");
  assert(callback(t, true, &e));
  // Malformed tensor shapes fail rather than silently skipping the intervention.
  r.direction = u;
  ggml_set_name(t, "ffn_out-1");
  t->ne[0] = 2;
  assert(!callback(t, false, &e));
  assert(e.failure.find("shape") != std::string::npos);
  ggml_free(ctx);
  // Validation failure on a context-less inspector must clear the pointer.
  float logits[1];
  assert(syn_research(&e, nullptr, 0, true, 1, 0, u, strength,
                      captured, stats, logits) < 0);
  assert(e.research == nullptr);
}

static void exercise_factors(ggml_backend_t backend, float strength) {
  Engine e;
  e.layers = 1;
  e.hidden = 3;
  e.inter = 4;
  e.total = 2;
  float u[] = {.6f, .8f, 0.f};
  float weights[] = {1, 2, 3, 4, -1, 0, 2, 3, 2, 1, -2, 4};
  float inputs[] = {1, -2, 3, .5f, 4, 3, 2, 1};
  float a[4] = {};
  for (int h = 0; h < 3; ++h)
    for (int k = 0; k < 4; ++k)
      a[k] += u[h] * weights[4*h+k];
  double stats[9] = {};
  Research r;
  r.direction = u;
  r.strength = strength;
  r.stats = stats;
  r.factors = a;
  r.offsets = {0, 0, 0};
  r.widths = {0, 4, 4};
  r.names = {"", "blk.0.ssm_out.weight", "blk.0.ffn_down.weight"};
  e.research = &r;
  auto *ctx = ggml_init({16384, nullptr, true});
  auto *w = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 3);
  auto *h = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 4, 2);
  ggml_set_name(w, "blk.0.ffn_down.weight");
  auto *y = ggml_mul_mat(ctx, w, h);
  ggml_set_name(y, "ffn_out-0");
  auto buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
  assert(buffer);
  ggml_backend_tensor_set(w, weights, 0, sizeof(weights));
  ggml_backend_tensor_set(h, inputs, 0, sizeof(inputs));
  float original[6] = {};
  for (int j = 0; j < 2; ++j)
    for (int row = 0; row < 3; ++row) {
      for (int k = 0; k < 4; ++k)
        original[j*3+row] += weights[row*4+k] * inputs[j*4+k];
      original[j*3+row] += .001f; // emulate independent quantized matmul error
    }
  ggml_backend_tensor_set(y, original, 0, sizeof(original));
  assert(callback(y, true, &e) && callback(y, false, &e));
  float actual[6];
  ggml_backend_tensor_get(y, actual, 0, sizeof(actual));
  for (int j = 0; j < 2; ++j) {
    double coefficient = 0;
    for (int k = 0; k < 4; ++k)
      coefficient += double(a[k]) * inputs[j*4+k];
    for (int row = 0; row < 3; ++row) {
      double expected = original[j*3+row] - strength * u[row] * coefficient;
      assert(std::abs(actual[j*3+row] - expected) < 1e-5);
    }
  }
  assert(stats[6] == 2);
  ggml_set_name(w, "wrong.writer.weight");
  assert(!callback(y, false, &e));
  assert(e.failure.find("Unsupported low-rank writer") != std::string::npos);
  ggml_backend_buffer_free(buffer);
  ggml_free(ctx);
}

int main(int argc, char **argv) {
  assert(argc == 2);
  ggml_backend_load_all_from_path(argv[1]);
  auto device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
  assert(device);
  auto backend = ggml_backend_dev_init(device, nullptr);
  assert(backend);
  for (float strength : {0.f, 0.25f, 1.f, -0.25f}) {
    exercise(backend, strength);
    exercise_factors(backend, strength);
  }
  for (float strength : {2.f, std::numeric_limits<float>::quiet_NaN()}) {
    bool rejected = false;
    float u[] = {1, 0, 0};
    try { synapse::validate_direction(u, 3, strength); }
    catch (const std::runtime_error &) { rejected = true; }
    assert(rejected);
  }
  float bad[] = {2, 0, 0};
  bool rejected = false;
  try { synapse::validate_direction(bad, 3, 0); }
  catch (const std::runtime_error &) { rejected = true; }
  assert(rejected);
  ggml_backend_free(backend);
  std::cout << "Native projection callback checks passed (synthetic CPU tensors, no inference).\n";
}

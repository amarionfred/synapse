// Synapse's bounded CPU tensor bridge. Build only against the pinned llama.cpp
// ABI. The callback stops the scheduler after ffn_swiglu-L, before ffn_down
// consumes it.
#include "llama.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

static thread_local std::string error;
struct Push {
  int layer, neuron, mode;
  float value;
}; // mode 0 add, 1 multiply
struct Engine {
  llama_model *model = nullptr;
  llama_context *ctx = nullptr;
  gguf_context *gguf = nullptr;
  ggml_context *tensors = nullptr;
  void *map = MAP_FAILED;
  size_t bytes = 0;
  int layers = 0, hidden = 0, inter = 0, vocab = 0, offset = 0, total = 0;
  int read_layer = -1, read_neuron = -1;
  float *acts = nullptr, *residuals = nullptr;
  bool full = false, observing = false;
  const Push *pushes = nullptr;
  int n_push = 0;
  std::vector<int> visits, resid_visits;
  std::string failure;
  ~Engine() {
    if (ctx)
      llama_free(ctx);
    if (model)
      llama_model_free(model);
    if (tensors)
      ggml_free(tensors);
    if (gguf)
      gguf_free(gguf);
    if (map != MAP_FAILED)
      munmap(map, bytes);
  }
  ggml_tensor *tensor(const std::string &name) {
    auto *t = ggml_get_tensor(tensors, name.c_str());
    if (!t)
      throw std::runtime_error("Missing GGUF tensor: " + name);
    return t;
  }
  const char *data(const ggml_tensor *t) {
    auto i = gguf_find_tensor(gguf, t->name);
    if (i < 0)
      throw std::runtime_error("Tensor absent from GGUF directory");
    size_t off = gguf_get_data_offset(gguf) + gguf_get_tensor_offset(gguf, i);
    if (off > bytes || ggml_nbytes(t) > bytes - off)
      throw std::runtime_error("Invalid tensor extent");
    return static_cast<char *>(map) + off;
  }
  void row_from(const ggml_tensor *t, const char *base, int r, float *out) {
    auto *traits = ggml_get_type_traits(t->type);
    auto *src = base + r * t->nb[1];
    if (t->type == GGML_TYPE_F32)
      memcpy(out, src, t->ne[0] * sizeof(float));
    else if (traits->to_float)
      traits->to_float(src, out, t->ne[0]);
    else
      throw std::runtime_error("No dequantizer for GGUF tensor");
  }
  void row(const ggml_tensor *t, int r, float *out) {
    row_from(t, data(t), r, out);
  }
  ggml_tensor *matrix(const std::string &name, int columns, int rows) {
    auto *t = tensor(name);
    if (t->ne[0] != columns || t->ne[1] != rows || t->ne[2] != 1 ||
        t->ne[3] != 1)
      throw std::runtime_error("Unexpected matrix shape: " + name);
    return t;
  }
  void column(int layer, int neuron, float *out) {
    if (layer < 0 || layer >= layers || neuron < 0 || neuron >= inter)
      throw std::runtime_error("Invalid neuron coordinate");
    auto *t = tensor("blk." + std::to_string(layer) + ".ffn_down.weight");
    if (t->ne[0] != inter || t->ne[1] != hidden)
      throw std::runtime_error("Unexpected down projection shape");
    auto *traits = ggml_get_type_traits(t->type);
    int block = ggml_blck_size(t->type);
    std::vector<float> buf(block);
    auto *src = data(t);
    for (int r = 0; r < hidden; ++r) {
      auto *at =
          src + r * t->nb[1] + (neuron / block) * ggml_type_size(t->type);
      if (t->type == GGML_TYPE_F32)
        memcpy(buf.data(), at, sizeof(float));
      else if (traits->to_float)
        traits->to_float(at, buf.data(), block);
      else
        throw std::runtime_error("Unsupported column quantizer");
      out[r] = buf[neuron % block];
    }
  }
};

static bool callback(ggml_tensor *t, bool ask, void *user) {
  auto &e = *static_cast<Engine *>(user);
  int layer = -1, consumed = 0;
  bool act = sscanf(t->name, "ffn_swiglu-%d%n", &layer, &consumed) == 1 &&
             t->name[consumed] == '\0';
  bool resid = !act && sscanf(t->name, "l_out-%d%n", &layer, &consumed) == 1 &&
               t->name[consumed] == '\0';
  bool push = false;
  for (int i = 0; act && i < e.n_push; ++i)
    push |= e.pushes[i].layer == layer;
  bool wanted = act && (e.full || e.read_layer == layer || push || e.observing);
  wanted |= resid && e.residuals;
  if (ask)
    return wanted;
  try {
    if (!wanted || layer < 0 || layer >= e.layers)
      throw std::runtime_error("Unexpected callback tensor");
    int width = act ? e.inter : e.hidden;
    int tokens = t->ne[1];
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) ||
        t->ne[0] != width || t->ne[2] != 1 || t->ne[3] != 1 ||
        e.offset + tokens > e.total)
      throw std::runtime_error("Unexpected tensor type, stride or shape at " +
                               std::string(t->name));
    std::vector<float> buf(size_t(width) * tokens);
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size() * sizeof(float));
    if (act) {
      e.visits[layer] += tokens;
      if (e.full)
        for (int j = 0; j < tokens; ++j)
          memcpy(e.acts + (size_t(layer) * e.total + e.offset + j) * width,
                 buf.data() + size_t(j) * width, width * sizeof(float));
      else if (e.acts && layer == e.read_layer)
        for (int j = 0; j < tokens; ++j)
          e.acts[e.offset + j] = buf[size_t(j) * width + e.read_neuron];
      if (push) {
        for (int i = 0; i < e.n_push; ++i)
          if (e.pushes[i].layer == layer) {
            const auto &p = e.pushes[i];
            for (int j = 0; j < tokens; ++j) {
              int begin = p.neuron < 0 ? 0 : p.neuron,
                  end = p.neuron < 0 ? width : p.neuron + 1;
              for (int n = begin; n < end; ++n) {
                float &v = buf[size_t(j) * width + n];
                v = p.mode ? v * p.value : v + p.value;
              }
            }
          }
        ggml_backend_tensor_set(t, buf.data(), 0, buf.size() * sizeof(float));
      }
    } else {
      e.resid_visits[layer] += tokens;
      for (int j = 0; j < tokens; ++j) {
        double sum = 0;
        for (int n = 0; n < width; ++n) {
          double v = buf[size_t(j) * width + n];
          sum += v * v;
        }
        e.residuals[size_t(layer) * e.total + e.offset + j] = std::sqrt(sum);
      }
    }
  } catch (const std::exception &ex) {
    e.failure = ex.what();
    return false;
  }
  return true;
}

extern "C" {
const char *syn_error() { return error.c_str(); }
void *syn_open(const char *path, const char *backend_dir, int threads,
               bool hooks) {
  try {
    ggml_backend_load_all_from_path(backend_dir);
    llama_backend_init();
    auto e = std::make_unique<Engine>();
    e->gguf = gguf_init_from_file(path, {true, &e->tensors});
    if (!e->gguf)
      throw std::runtime_error("Cannot read GGUF");
    auto arch = gguf_find_key(e->gguf, "general.architecture");
    if (arch < 0 || gguf_get_kv_type(e->gguf, arch) != GGUF_TYPE_STRING ||
        std::string(gguf_get_val_str(e->gguf, arch)) != "qwen35")
      throw std::runtime_error(
          "Only the qwen35 GGUF graph has been validated by this bridge");
    int fd = open(path, O_RDONLY);
    struct stat st{};
    if (fd < 0)
      throw std::runtime_error("Cannot open GGUF weights");
    if (fstat(fd, &st)) {
      close(fd);
      throw std::runtime_error("Cannot stat GGUF");
    }
    e->bytes = st.st_size;
    e->map = mmap(nullptr, e->bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (e->map == MAP_FAILED)
      throw std::runtime_error("Cannot map GGUF weights");
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    e->model = llama_model_load_from_file(path, mp);
    if (!e->model)
      throw std::runtime_error("llama.cpp failed to load weights");
    e->layers = llama_model_n_layer(e->model);
    e->hidden = llama_model_n_embd(e->model);
    e->inter = e->tensor("blk.0.ffn_down.weight")->ne[0];
    e->vocab = llama_vocab_n_tokens(llama_model_get_vocab(e->model));
    auto cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.cb_eval = hooks ? callback : nullptr;
    cp.cb_eval_user_data = e.get();
    e->ctx = llama_init_from_model(e->model, cp);
    if (!e->ctx)
      throw std::runtime_error("Cannot allocate llama context");
    return e.release();
  } catch (const std::exception &ex) {
    error = ex.what();
    return nullptr;
  }
}
void syn_close(void *ptr) { delete static_cast<Engine *>(ptr); }
int syn_vocab(void *ptr) { return static_cast<Engine *>(ptr)->vocab; }
int syn_tokenize(void *ptr, const char *text, int len, int *out, int cap,
                 bool special) {
  return llama_tokenize(
      llama_model_get_vocab(static_cast<Engine *>(ptr)->model), text, len, out,
      cap, special, true);
}
int syn_piece(void *ptr, int token, char *out, int cap, bool special) {
  return llama_token_to_piece(
      llama_model_get_vocab(static_cast<Engine *>(ptr)->model), token, out, cap,
      0, special);
}
bool syn_eog(void *ptr, int token) {
  return llama_vocab_is_eog(
      llama_model_get_vocab(static_cast<Engine *>(ptr)->model), token);
}
int syn_eval(void *ptr, int *ids, int count, bool reset, bool observe,
             bool full, int layer, int neuron, const Push *pushes, int n_push,
             float *acts, float *residuals, float *logits) {
  auto &e = *static_cast<Engine *>(ptr);
  try {
    if (count < 1 || count > 2048)
      throw std::runtime_error("Probe needs 1..2048 tokens");
    if (acts && !full &&
        (layer < 0 || layer >= e.layers || neuron < 0 || neuron >= e.inter))
      throw std::runtime_error("Invalid read coordinate");
    for (int i = 0; i < n_push; ++i)
      if (pushes[i].layer < 0 || pushes[i].layer >= e.layers ||
          pushes[i].neuron < -1 || pushes[i].neuron >= e.inter ||
          (pushes[i].neuron == -1 && pushes[i].mode != 1) ||
          (pushes[i].mode != 0 && pushes[i].mode != 1) ||
          !std::isfinite(pushes[i].value))
        throw std::runtime_error("Invalid intervention");
    e.total = count;
    e.acts = acts;
    e.residuals = residuals;
    e.full = full;
    e.observing = observe;
    e.read_layer = layer;
    e.read_neuron = neuron;
    e.pushes = pushes;
    e.n_push = n_push;
    e.visits.assign(e.layers, 0);
    e.resid_visits.assign(e.layers, 0);
    e.failure.clear();
    if (reset)
      llama_memory_clear(llama_get_memory(e.ctx), true);
    for (int off = 0; off < count; off += 64) {
      e.offset = off;
      int n = std::min(64, count - off);
      auto batch = llama_batch_get_one(ids + off, n);
      // All logits preserve all token rows even in the last layer during
      // inspection.
      std::vector<int8_t> outputs(n, 1);
      if (acts || residuals || observe)
        batch.logits = outputs.data();
      int rc = llama_decode(e.ctx, batch);
      if (!e.failure.empty())
        throw std::runtime_error(e.failure);
      if (rc)
        throw std::runtime_error("llama_decode returned " + std::to_string(rc));
    }
    for (int l = 0; l < e.layers; ++l) {
      if ((full || observe || (acts && l == layer)) && e.visits[l] != count)
        throw std::runtime_error("Missing MLP callback at layer " +
                                 std::to_string(l));
      if (residuals && e.resid_visits[l] != count)
        throw std::runtime_error("Missing residual callback at layer " +
                                 std::to_string(l));
      for (int i = 0; i < n_push; ++i)
        if (pushes[i].layer == l && !e.visits[l])
          throw std::runtime_error("Intervention hook did not execute");
    }
    memcpy(logits, llama_get_logits_ith(e.ctx, -1),
           size_t(e.vocab) * sizeof(float));
    e.pushes = nullptr;
    e.n_push = 0;
    e.acts = e.residuals = nullptr;
    return 0;
  } catch (const std::exception &ex) {
    error = ex.what();
    e.pushes = nullptr;
    e.n_push = 0;
    e.acts = e.residuals = nullptr;
    llama_memory_clear(llama_get_memory(e.ctx), true);
    return -1;
  }
}
int syn_column(void *ptr, int layer, int neuron, float *out) {
  try {
    static_cast<Engine *>(ptr)->column(layer, neuron, out);
    return 0;
  } catch (const std::exception &ex) {
    error = ex.what();
    return -1;
  }
}
// Bounded row export for batched lens scoring and independent matrix checks.
int syn_rows(void *ptr, const char *name, int start, int count, int width,
             float *out) {
  try {
    auto &e = *static_cast<Engine *>(ptr);
    auto *t = ggml_get_tensor(e.tensors, name);
    if (!t && std::string(name) == "output.weight")
      t = e.tensor("token_embd.weight");
    if (!t || start < 0 || count < 1 || count > 2048 ||
        int64_t(start) + count > t->ne[1] || width != t->ne[0] ||
        t->ne[2] != 1 || t->ne[3] != 1)
      throw std::runtime_error("Invalid weight-row request");
    const char *base = e.data(t);
    for (int r = 0; r < count; ++r)
      e.row_from(t, base, start + r, out + size_t(r) * width);
    return 0;
  } catch (const std::exception &ex) {
    error = ex.what();
    return -1;
  }
}
// Exact float dot products of the dequantized GGUF weights, matching the
// Transformers direct-path heuristic. Each output is [layers, neurons].
// Norm gains in GGUF already include Qwen's +1 conversion; never add 1 again.
int syn_connections(void *ptr, int layer, int neuron, int size, float *gate_out,
                    float *up_out) {
  try {
    auto &e = *static_cast<Engine *>(ptr);
    if (size != e.layers * e.inter)
      throw std::runtime_error("Invalid connection output capacity");
    std::vector<float> d(e.hidden), gain(e.hidden), rg(e.hidden), ru(e.hidden);
    std::vector<float> row(std::max(e.inter, e.hidden)), v(e.hidden);
    e.column(layer, neuron, d.data()); // validates the coordinate first
    const auto prefix = "blk." + std::to_string(layer) + ".";
    e.row(e.matrix(prefix + "post_attention_norm.weight", e.hidden, 1), 0,
          gain.data());
    e.row(e.matrix(prefix + "ffn_gate.weight", e.hidden, e.inter), neuron,
          rg.data());
    e.row(e.matrix(prefix + "ffn_up.weight", e.hidden, e.inter), neuron,
          ru.data());
    for (int h = 0; h < e.hidden; ++h) {
      rg[h] *= gain[h];
      ru[h] *= gain[h];
    }
    std::fill(gate_out, gate_out + size, 0.f);
    std::fill(up_out, up_out + size, 0.f);
    for (int l = 0; l < e.layers; ++l) {
      if (l == layer)
        continue;
      const auto p = "blk." + std::to_string(l) + ".";
      float *g = gate_out + size_t(l) * e.inter,
            *u = up_out + size_t(l) * e.inter;
      if (l > layer) {
        e.row(e.matrix(p + "post_attention_norm.weight", e.hidden, 1), 0,
              gain.data());
        for (int h = 0; h < e.hidden; ++h)
          v[h] = d[h] * gain[h];
        auto *gt = e.matrix(p + "ffn_gate.weight", e.hidden, e.inter);
        auto *ut = e.matrix(p + "ffn_up.weight", e.hidden, e.inter);
        const char *gb = e.data(gt), *ub = e.data(ut);
        for (int n = 0; n < e.inter; ++n) {
          e.row_from(gt, gb, n, row.data());
          double dot = 0;
          for (int h = 0; h < e.hidden; ++h)
            dot += double(row[h]) * v[h];
          g[n] = dot;
          e.row_from(ut, ub, n, row.data());
          dot = 0;
          for (int h = 0; h < e.hidden; ++h)
            dot += double(row[h]) * v[h];
          u[n] = dot;
        }
      } else {
        auto *dt = e.matrix(p + "ffn_down.weight", e.inter, e.hidden);
        const char *db = e.data(dt);
        std::vector<double> gg(e.inter, 0.), uu(e.inter, 0.);
        for (int h = 0; h < e.hidden; ++h) {
          e.row_from(dt, db, h, row.data());
          for (int n = 0; n < e.inter; ++n) {
            gg[n] += double(row[n]) * rg[h];
            uu[n] += double(row[n]) * ru[h];
          }
        }
        for (int n = 0; n < e.inter; ++n) {
          g[n] = gg[n];
          u[n] = uu[n];
        }
      }
    }
    return 0;
  } catch (const std::exception &ex) {
    error = ex.what();
    return -1;
  }
}
int syn_lens(void *ptr, int layer, int neuron, float *out) {
  try {
    auto &e = *static_cast<Engine *>(ptr);
    std::vector<float> d(e.hidden), norm(e.hidden);
    e.column(layer, neuron, d.data());
    auto *nt = e.tensor("output_norm.weight");
    if (nt->ne[0] != e.hidden)
      throw std::runtime_error("Unexpected final norm shape");
    e.row(nt, 0, norm.data());
    double sum = 0;
    for (float v : d)
      sum += double(v) * v;
    int k = gguf_find_key(e.gguf, "qwen35.attention.layer_norm_rms_epsilon");
    float eps = k < 0 ? 1e-6f : gguf_get_val_f32(e.gguf, k);
    float scale = 1 / std::sqrt(sum / e.hidden + eps);
    for (int j = 0; j < e.hidden; ++j)
      d[j] *= scale * norm[j];
    auto *head = ggml_get_tensor(e.tensors, "output.weight");
    if (!head)
      head = e.tensor("token_embd.weight");
    if (head->ne[0] != e.hidden || head->ne[1] != e.vocab)
      throw std::runtime_error("Unexpected unembedding shape");
    // Dequantize one row at a time; never materialize the whole FP32 model.
    std::vector<float> row(e.hidden);
    for (int i = 0; i < e.vocab; ++i) {
      e.row(head, i, row.data());
      double dot = 0;
      for (int j = 0; j < e.hidden; ++j)
        dot += double(row[j]) * d[j];
      out[i] = dot;
    }
    return 0;
  } catch (const std::exception &ex) {
    error = ex.what();
    return -1;
  }
}
}

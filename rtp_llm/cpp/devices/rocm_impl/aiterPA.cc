#include "rtp_llm/cpp/devices/rocm_impl/aiterPA.h"
#include "rtp_llm/cpp/devices/rocm_impl/atrexPA.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/rocm_impl/ROCmDevice.h"
#include "rtp_llm/cpp/devices/utils/DebugUtils.h"
#include <filesystem>
#include <set>
#include <unordered_map>

using namespace pybind11::literals;
namespace fs = std::filesystem;

namespace rtp_llm {

template <typename ReturnType = void, typename... Args>
ReturnType call_func_ptr(void* func_ptr, Args... args){
    auto func = reinterpret_cast<ReturnType (*)(Args...)>(func_ptr);
    return func(std::forward<Args>(args)...);
}

std::string get_pa_compile_dtype(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kBFloat16:
            return "__hip_bfloat16";
        case torch::kFloat16:
            return "_Float16";
        case torch::kFloat8_e4m3fnuz:
        case torch::kFloat8_e4m3fn:
            return "uint8_t";
        default:
            throw std::runtime_error("Unsupported dtype");
    }
}

static inline uint64_t next_power_of_2(uint64_t n) {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    return n + 1;
}

static int get_cdna_version() {
    static int cdna_version = []() {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        std::string gcn_arch = prop.gcnArchName;
        if (gcn_arch.find("gfx950") != std::string::npos) {
            return 4;
        } else if (gcn_arch.find("gfx942") != std::string::npos) {
            return 3;
        }
        return -1;
    }();
    return cdna_version;
}

AiterWrapper::AiterWrapper(const DeviceInitParams& params) {
    if (!Py_IsInitialized()) {
        return;
    }
    py::gil_scoped_acquire acquire;
    pa_gluon_aot_api = py::module_::import("aiter_meta.csrc.cpp_itfs.pa_gluon_aot.api");
    pa_gluon_load_libs = pa_gluon_aot_api.attr("load_all_libs");
    hip_pa_api = py::module_::import("aiter_meta.csrc.cpp_itfs.pa.pa_api");
    hip_pa_load_libs = hip_pa_api.attr("load_all_libs");
}

void AiterWrapper::runTritonPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_mtp, hipStream_t stream) {
    py::gil_scoped_acquire acquire;
    size_t  token_num  = params.input.shape()[0];
    size_t  num_heads  = params.configs.head_num;
    size_t  head_size  = params.configs.size_per_head;

    bool prefill_pa = (params.common.sequence_lengths != nullptr &&
                      params.common.sequence_lengths->data() == nullptr);
    int64_t partition_size = 256;
    int64_t mtp = prefill_pa ? params.common.context_max_seq_len : 1;
    int64_t num_kv_heads = params.configs.kv_head_num;
    int64_t max_seq_len = prefill_pa ?
                (params.common.context_max_seq_len + params.common.max_prefix_length) :
                (device->nativeGraphCapturing() ? device->initParams().max_seq_len : params.common.decoder_max_seq_len + 1);

    size_t query_group_size = num_heads / (size_t)num_kv_heads;
    size_t multi_query_group_size = num_heads / (size_t)num_kv_heads * (size_t)mtp;
    size_t max_num_partitions = (max_seq_len + partition_size - 1) / partition_size;

    size_t batch_size = prefill_pa ? params.common.context_batch_size : params.common.decoder_batch_size;

    BufferPtr exp_sums_buffer = device->allocateBuffer({rtp_llm::DataType::TYPE_FP32,
            {batch_size, (size_t)num_kv_heads, max_num_partitions, multi_query_group_size}, AllocationType::DEVICE}, {"exp_sums"});
    BufferPtr max_logits_buffer = device->allocateBuffer({rtp_llm::DataType::TYPE_FP32,
            {batch_size, (size_t)num_kv_heads, max_num_partitions, multi_query_group_size}, AllocationType::DEVICE}, {"max_logits"});
    BufferPtr tmp_out_buffer = device->allocateBuffer({params.output.type(),
            {batch_size, (size_t)num_kv_heads, max_num_partitions, multi_query_group_size, head_size}, AllocationType::DEVICE}, {"tmp_out"});

    auto out                   = Buffer2torchTensor(params.output, false);
    auto exp_sums              = Buffer2torchTensor(exp_sums_buffer, false);
    auto max_logits            = Buffer2torchTensor(max_logits_buffer, false);
    auto tmp_out               = Buffer2torchTensor(tmp_out_buffer, false);
    auto query                 = Buffer2torchTensor(q_mtp, false);
    auto key_cache             = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 0);
    auto value_cache           = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 1);
    auto block_tables          = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id, false);

    auto seq_lens = prefill_pa ? Buffer2torchTensor(params.common.kv_seqlens, false) :
                            ((AiterAttnParams*)params.common.decode_aiter_attn.get())->sequence_lengths_t;
    out = out.view(query.sizes());

    float scale = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);

    int64_t x = 16 / key_cache.element_size();
    auto kv_sizes = key_cache.sizes();
    // k_cache [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    // v_cache [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    key_cache = key_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3] / x, kv_sizes[2], x});
    value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[2] / x, kv_sizes[3], x});

    torch::Tensor q_quant, q_scale, query_scale_gluon;
    torch::Tensor k_scale, v_scale;

    int kv_quant_mode = -1;
    std::string kv_cache_dtype = "auto";
    if (key_cache.dtype() == at::kFloat8_e4m3fnuz) {
        kv_cache_dtype = "fp8";
        k_scale = Buffer2torchTensor(params.common.kv_cache->kv_scale_buffer, false);
        v_scale = k_scale;
        if (k_scale.numel() > 1) {
            kv_quant_mode = 1;
            k_scale.unsqueeze_(-1);
            v_scale.unsqueeze_(-1);
        } else {
            kv_quant_mode = 0;
        }
    }

    std::optional<torch::Tensor> fp8_out_scale = std::nullopt;
    std::optional<torch::Tensor> alibi_slopes;

    torch::Tensor output_gluon, query_gluon;
    std::vector<void *> pa_decode_gluon_ptrs;
    {
        py::gil_scoped_acquire acquire;
        std::vector<unsigned long> py_list = pa_gluon_load_libs(
                    "bfloat16",         // data_type
                    head_size,          // last_dim
                    mtp,                // query_length
                    num_heads,          // num_query_heads
                    num_kv_heads,       // num_kv_heads
                    head_size,          // head_size
                    key_cache.size(3),  // kv_block_size
                    partition_size,     // context_partition_size
                    -1,                 // query_quant_mode
                    kv_quant_mode,      // kv_quant_mode
                    kv_cache_dtype,     // kv_cache_dtype
                    true,               // value_transposed
                    0,                  // use_sinks
                    get_cdna_version()  // cdna_version
                ).cast<std::vector<unsigned long>>();
        pa_decode_gluon_ptrs.reserve(py_list.size());
        for (unsigned long item : py_list) {
            pa_decode_gluon_ptrs.push_back((void *)item);
        }
    }

    if (mtp > 1) {
        auto stride_input_batch   = mtp * num_heads * head_size;
        auto stride_input_seq     = num_heads * head_size;
        auto stride_input_head    = query_group_size * head_size;

        auto stride_input_group   = head_size;
        auto stride_output_batch  = num_kv_heads * mtp * query_group_size * head_size;
        auto stride_output_merged = head_size;

        auto merged_dim_size = num_kv_heads * mtp * query_group_size;
        auto merged_block_size = next_power_of_2(merged_dim_size);
        auto block_size_last = next_power_of_2(head_size);
        auto grid_dim_0 = batch_size;
        auto grid_dim_1 = (merged_dim_size + merged_block_size - 1) / merged_block_size;;
        auto grid_dim_2 = (head_size + block_size_last - 1) / block_size_last;

        output_gluon = torch::empty({(int)batch_size, (int)num_kv_heads * (int)mtp * (int)query_group_size, (int)head_size},
                                        torch::TensorOptions().dtype(out.dtype()).device(torch::Device(torch::kCUDA)));
        query_gluon = torch::empty({(int)batch_size, (int)num_kv_heads * (int)mtp * (int)query_group_size, (int)head_size},
                                        torch::TensorOptions().dtype(query.dtype()).device(torch::Device(torch::kCUDA)));

        auto output_gluon_sizes = output_gluon.sizes();
        auto exp_sums_sizes = exp_sums.sizes();
        auto tmp_out_sizes = tmp_out.sizes();
        auto query_gluon_sizes = query_gluon.sizes();
        auto key_cache_sizes = key_cache.sizes();
        auto value_cache_sizes = value_cache.sizes();
        auto block_tables_sizes = block_tables.sizes();

        //pa_decode_gluon_ptrs[0]: transpose_query_gluon_kernel
        call_func_ptr<void>(pa_decode_gluon_ptrs[0],
                            query.data_ptr(),
                            query_gluon.data_ptr(),
                            batch_size, mtp, num_kv_heads, query_group_size, head_size,
                            stride_input_batch, stride_input_seq, stride_input_head, stride_input_group,
                            stride_output_batch, stride_output_merged,
                            grid_dim_0, grid_dim_1, grid_dim_2,stream);

        //pa_decode_gluon_ptrs[1]: pa_decode_attention_reduce_kernel
        call_func_ptr<void>(pa_decode_gluon_ptrs[1],
                            output_gluon.data_ptr(),
                            exp_sums.data_ptr(),
                            max_logits.data_ptr(),
                            tmp_out.data_ptr(),
                            query_gluon.data_ptr(),
                            key_cache.data_ptr(),
                            value_cache.data_ptr(),
                            block_tables.data_ptr(),
                            seq_lens.data_ptr(),
                            nullptr,
                            scale,
                            nullptr,
                            k_scale.defined()? k_scale.data_ptr() : nullptr,
                            v_scale.defined()? v_scale.data_ptr() : nullptr,
                            output_gluon.stride(0), output_gluon.stride(1),
                            exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
                            tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2), tmp_out.stride(3),
                            query_gluon.stride(0), query_gluon.stride(1),
                            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                            block_tables.stride(0),
                            0,
                            k_scale.defined()? k_scale.stride(0):0,
                            k_scale.defined()? k_scale.stride(1):0,
                            batch_size, num_kv_heads, max_num_partitions,
                            mtp, query_group_size, multi_query_group_size, head_size, stream);

        auto output_stride_input_batch = num_kv_heads * mtp * query_group_size * head_size;
        auto output_stride_input_kv_head = mtp * query_group_size * head_size;
        auto output_stride_input_seq = query_group_size * head_size;
        auto output_stride_input_group = head_size;
        auto output_stride_output_batch_seq = num_heads * head_size;
        auto output_stride_output_merged = head_size;

        auto output_merged_dim_size = num_kv_heads * query_group_size;
        auto output_merged_block_size = next_power_of_2(output_merged_dim_size);
        auto output_block_size_last = next_power_of_2(head_size);

        auto output_grid_dim_0 = batch_size * mtp;
        auto output_grid_dim_1 = (output_merged_dim_size + output_merged_block_size - 1) / output_merged_block_size;;
        auto output_grid_dim_2 = (head_size + output_block_size_last - 1) / output_block_size_last;

        //pa_decode_gluon_ptrs[2]: transpose_output_gluon_kernel
        call_func_ptr<void>(pa_decode_gluon_ptrs[2],
                            output_gluon.data_ptr(),
                            out.data_ptr(),
                            batch_size,
                            mtp,
                            num_kv_heads,
                            query_group_size,
                            head_size,
                            output_stride_input_batch,
                            output_stride_input_kv_head,
                            output_stride_input_seq,
                            output_stride_input_group,
                            output_stride_output_batch_seq,
                            output_stride_output_merged,
                            output_grid_dim_0,
                            output_grid_dim_1,
                            output_grid_dim_2,
                            stream);
    } else {
        //pa_decode_gluon_ptrs[1]: pa_decode_attention_reduce_kernel
        call_func_ptr<void>(pa_decode_gluon_ptrs[1],
                            out.data_ptr(),
                            exp_sums.data_ptr(),
                            max_logits.data_ptr(),
                            tmp_out.data_ptr(),
                            query.data_ptr(),
                            key_cache.data_ptr(),
                            value_cache.data_ptr(),
                            block_tables.data_ptr(),
                            seq_lens.data_ptr(),
                            nullptr,
                            scale,
                            nullptr,
                            k_scale.defined()? k_scale.data_ptr() : nullptr,
                            v_scale.defined()? v_scale.data_ptr() : nullptr,
                            out.stride(0), out.stride(1),
                            exp_sums.stride(0), exp_sums.stride(1), exp_sums.stride(2),
                            tmp_out.stride(0), tmp_out.stride(1), tmp_out.stride(2), tmp_out.stride(3),
                            query.stride(0), query.stride(1),
                            key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
                            value_cache.stride(0), value_cache.stride(1), value_cache.stride(2),
                            block_tables.stride(0),
                            0,
                            k_scale.defined()? k_scale.stride(0):0,
                            k_scale.defined()? k_scale.stride(1):0,
                            batch_size, num_kv_heads, max_num_partitions,
                            mtp, query_group_size, multi_query_group_size, head_size, stream);
    }
}

void AiterWrapper::runHipPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_tmp, hipStream_t stream) {
    py::gil_scoped_acquire acquire;
    size_t  token_num  = params.input.shape()[0];
    size_t  num_heads  = params.configs.head_num;
    size_t  head_size  = params.configs.size_per_head;

    bool prefill_pa = (params.common.sequence_lengths != nullptr &&
                      params.common.sequence_lengths->data() == nullptr);

    int64_t warp_size = 64;
    int64_t partition_size = 256;
    int64_t q_length = prefill_pa ? params.common.context_max_seq_len : 1;
    int64_t num_kv_heads = params.configs.kv_head_num;
    int64_t max_seq_len = prefill_pa ?
                (params.common.context_max_seq_len + params.common.max_prefix_length) :
                (device->nativeGraphCapturing() ? device->initParams().max_seq_len : params.common.decoder_max_seq_len + 1);

    size_t query_group_size = num_heads / (size_t)num_kv_heads;
    size_t max_num_partitions = (max_seq_len + partition_size - 1) / partition_size;
    size_t npar_loops = (max_num_partitions + warp_size - 1) / warp_size;
    size_t batch_size = prefill_pa ? params.common.context_batch_size : params.common.decoder_batch_size;

    BufferPtr exp_sums_buffer = device->allocateBuffer({rtp_llm::DataType::TYPE_FP32,
            {token_num, num_heads, max_num_partitions}, AllocationType::DEVICE}, {"exp_sums"});

    BufferPtr max_logits_buffer = device->allocateBuffer({rtp_llm::DataType::TYPE_FP32,
            {token_num, num_heads, max_num_partitions}, AllocationType::DEVICE}, {"max_logits"});

    BufferPtr tmp_out_buffer = device->allocateBuffer({params.output.type(),
            {token_num, num_heads, max_num_partitions, head_size}, AllocationType::DEVICE}, {"tmp_out"});

    auto out          = Buffer2torchTensor(params.output, false);
    auto exp_sums     = Buffer2torchTensor(exp_sums_buffer, false);
    auto max_logits   = Buffer2torchTensor(max_logits_buffer, false);
    auto tmp_out      = Buffer2torchTensor(tmp_out_buffer, false);
    auto query        = Buffer2torchTensor(q_tmp, false);
    auto key_cache    = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 0);
    auto value_cache  = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 1);

    float scale = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);
    auto block_tables = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id, false);
    auto seq_lens = prefill_pa ? Buffer2torchTensor(params.common.kv_seqlens, false) :
                            ((AiterAttnParams*)params.common.decode_aiter_attn.get())->sequence_lengths_t;

    size_t max_num_blocks = block_tables.size(1);
    int64_t block_size = params.configs.tokens_per_block;

    bool v_shuffle = true;
    int64_t x = 16 / key_cache.element_size();
    auto kv_sizes = value_cache.sizes();
    key_cache = key_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3] / x, kv_sizes[2], x});
    value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[2] / x, kv_sizes[3], x});

    torch::Tensor q_quant, q_scale;
    torch::Tensor k_scale, v_scale;

    std::string kv_cache_dtype = "auto";
    std::string quant_method = "vllm::Fp8QuantMethod::kPerTensor";

    if (key_cache.dtype() == at::kFloat8_e4m3fnuz) {
        kv_cache_dtype = "fp8";
        quant_method = "vllm::Fp8QuantMethod::kPerHead";
        k_scale = Buffer2torchTensor(params.common.kv_cache->kv_scale_buffer, false);
        v_scale = k_scale;
    }

    std::optional<torch::Tensor> fp8_out_scale = std::nullopt;
    std::optional<torch::Tensor> alibi_slopes;

    void* paged_attention_rocm_ptr;
    {
        py::gil_scoped_acquire acquire;
        paged_attention_rocm_ptr = (void*)hip_pa_load_libs(
            query_group_size,
            head_size,
            npar_loops,
            get_pa_compile_dtype(query.scalar_type()),
            get_pa_compile_dtype(key_cache.scalar_type()),
            kv_cache_dtype,
            get_pa_compile_dtype(out.scalar_type()),
            block_size,
            alibi_slopes.has_value() ? "true" : "false",
            q_length, //query_length
            quant_method,
            v_shuffle).cast<unsigned long>();
    }

    call_func_ptr<void>(paged_attention_rocm_ptr,
                        out.data_ptr(),
                        exp_sums.data_ptr(),
                        max_logits.data_ptr(),
                        tmp_out.data_ptr(),
                        query.data_ptr(),
                        key_cache.data_ptr(),
                        value_cache.data_ptr(),
                        scale,
                        block_tables.data_ptr(),
                        seq_lens.data_ptr(),
                        max_seq_len,
                        batch_size,
                        num_kv_heads,
                        num_heads,
                        max_num_blocks,
                        query.stride(0),
                        key_cache.stride(0),
                        key_cache.stride(1),
                        nullptr,        //alibi_slopes
                        q_scale.defined() ? q_scale.data_ptr() : nullptr,
                        k_scale.defined() ? k_scale.data_ptr() : nullptr,
                        v_scale.defined() ? v_scale.data_ptr() : nullptr,
                        nullptr,        //fp8_out_scale_ptr
                        stream);

}

inline torch::Tensor Buffer2torchTensorCustom(const Buffer& buf, std::vector<int64_t> shape, size_t offset = 0) {
    auto option =
        torch::dtype(dataTypeToTorchType(buf.type())).device(memoryTypeToTorchDevice(buf.where())).requires_grad(false);
    return torch::from_blob((void*)((char*)(buf.data()) + offset), shape, option);
}

void runAiterAsmPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_tmp, bool dump_pa_inputs) {
    auto out   = Buffer2torchTensor(params.output, false);
    auto query = Buffer2torchTensor(q_tmp, false);

    if (q_tmp.shape().size() < 3) {
        throw std::runtime_error("aiter_paged_attention only support 3-dim input");
    } else if (q_tmp.shape().size() > 3) {
        query = query.reshape({query.size(0), query.size(1), -1});
    }

    size_t  num_heads      = params.configs.head_num;
    int64_t partition_size = 256;
    int64_t max_seq_len    = params.common.decoder_max_seq_len + 1;

    auto key_cache   = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 0);
    auto value_cache = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 1);

    auto block_tables = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id, false);

    auto context_lens = Buffer2torchTensor(params.common.sequence_lengths, false);
    context_lens      = context_lens + 1;

    int                          max_num_blocks = block_tables.size(1);
    std::optional<torch::Tensor> K_QScale       = std::nullopt;
    std::optional<torch::Tensor> V_QScale       = std::nullopt;
    std::optional<torch::Tensor> out_opt        = out;
    
    if (dump_pa_inputs) {
        static std::unordered_map<int64_t, int> pa_step_counter;
        static std::set<std::string> layer_requests_saved;
        static std::mutex pa_step_mutex;
        static int last_saved_round = -1;
        
        auto req_id_host = device->clone({*params.common.decode_request_id, AllocationType::HOST});
        auto bt_host = device->clone({*params.common.kv_cache->kv_cache_block_id, AllocationType::HOST});
        auto sl_host = device->clone({*params.common.sequence_lengths, AllocationType::HOST});
        ((ROCmDevice*)device)->syncAndCheck();
        
        const int64_t* req_ids = req_id_host->data<int64_t>();
        const int32_t* bt_data = bt_host->data<int32_t>();
        const int32_t* sl_data = sl_host->data<int32_t>();
        size_t batch_size = params.common.decoder_batch_size;
        size_t bt_cols = bt_host->shape().size() > 1 ? bt_host->shape()[1] : 1;
        
        int round = static_cast<int>(req_ids[0] / 1000000);
        int layer_id = params.layer_id;
        
        if (round != last_saved_round) {
            layer_requests_saved.clear();
            last_saved_round = round;
        }
        
        bool should_save_layer = (layer_id == 0 || layer_id == 1 || layer_id == 13 || layer_id == 27);
        
        bool need_save_blocks = false;
        if (should_save_layer) {
            for (size_t i = 0; i < batch_size; i++) {
                std::string key = std::to_string(layer_id) + "_" + std::to_string(req_ids[i]);
                if (layer_requests_saved.find(key) == layer_requests_saved.end()) {
                    need_save_blocks = true;
                    break;
                }
            }
        }
        
        if (should_save_layer && need_save_blocks) {
            std::string kv_dir = "/mnt/raid0/yilin/rtp-llm/zztmp/pa_inputs/round" + std::to_string(round) + "/layer" + std::to_string(layer_id);
            if (!fs::exists(kv_dir)) {
                fs::create_directories(kv_dir);
            }
            
            auto kv_buffer = params.common.kv_cache->kv_cache_buffer;
            ((ROCmDevice*)device)->syncAndCheck();
            
            saveBufferDataToTorch(*bt_host, nullptr, kv_dir + "/block_tables.pt");
            saveBufferDataToTorch(*sl_host, nullptr, kv_dir + "/sequence_lengths.pt");
            
            std::set<int> all_used_blocks;
            int total_saved_blocks = 0;
            int new_requests_count = 0;
            
            for (size_t i = 0; i < batch_size; i++) {
                int64_t req_id = req_ids[i];
                
                std::string layer_req_key = std::to_string(layer_id) + "_" + std::to_string(req_id);
                if (layer_requests_saved.find(layer_req_key) != layer_requests_saved.end()) {
                    continue;
                }
                
                new_requests_count++;
                
                std::string req_dir = kv_dir + "/req" + std::to_string(req_id);
                fs::create_directories(req_dir);
                
                int32_t seq_len = sl_data[i];
                int tokens_per_block = 16;
                int valid_blocks = (seq_len + tokens_per_block) / tokens_per_block;
                int start_block = std::max(0, valid_blocks - 3);
                
                std::vector<int> saved_blocks;
                int req_block_count = 0;
                
                for (int b = start_block; b < valid_blocks && b < (int)bt_cols; b++) {
                    int block_id = bt_data[i * bt_cols + b];
                    if (block_id < 0) break;
                    
                    all_used_blocks.insert(block_id);
                    
                    if (block_id < (int)kv_buffer->shape()[0]) {
                        BufferPtr block = kv_buffer->index(block_id);
                        auto block_host = device->clone({*block, AllocationType::HOST});
                        ((ROCmDevice*)device)->syncAndCheck();
                        
                        if (block_host->dim() > 0 && block_host->shape()[0] >= 2) {
                            BufferPtr k_block = block_host->index(0);
                            BufferPtr v_block = block_host->index(1);
                            saveBufferDataToTorch(*k_block, nullptr, req_dir + "/block" + std::to_string(block_id) + "_K.pt");
                            saveBufferDataToTorch(*v_block, nullptr, req_dir + "/block" + std::to_string(block_id) + "_V.pt");
                            saved_blocks.push_back(block_id);
                            req_block_count++;
                            total_saved_blocks++;
                        }
                    }
                }
                
                FILE* req_fp = fopen((req_dir + "/block_info.txt").c_str(), "w");
                if (req_fp) {
                    fprintf(req_fp, "request_id: %ld\n", req_id);
                    fprintf(req_fp, "layer_id: %d\n", layer_id);
                    fprintf(req_fp, "seq_len: %d\n", seq_len);
                    fprintf(req_fp, "valid_blocks: %d\n", valid_blocks);
                    fprintf(req_fp, "saved_blocks (last 3): [");
                    for (size_t sb = 0; sb < saved_blocks.size(); sb++) {
                        fprintf(req_fp, "%d%s", saved_blocks[sb], sb + 1 < saved_blocks.size() ? "," : "");
                    }
                    fprintf(req_fp, "]\n");
                    fclose(req_fp);
                }
                
                layer_requests_saved.insert(layer_req_key);
            }
            
            FILE* kv_fp = fopen((kv_dir + "/kv_cache_info.txt").c_str(), "w");
            if (kv_fp) {
                fprintf(kv_fp, "round: %d\n", round);
                fprintf(kv_fp, "layer_id: %d\n", layer_id);
                fprintf(kv_fp, "kv_cache_buffer_shape: [");
                for (size_t d = 0; d < kv_buffer->shape().size(); d++) {
                    fprintf(kv_fp, "%zu%s", kv_buffer->shape()[d], d + 1 < kv_buffer->shape().size() ? ", " : "");
                }
                fprintf(kv_fp, "]\n");
                fprintf(kv_fp, "block_tables_shape: [");
                for (size_t d = 0; d < bt_host->shape().size(); d++) {
                    fprintf(kv_fp, "%zu%s", bt_host->shape()[d], d + 1 < bt_host->shape().size() ? ", " : "");
                }
                fprintf(kv_fp, "]\n");
                fprintf(kv_fp, "batch_size: %zu\n", batch_size);
                fprintf(kv_fp, "total_saved_blocks: %d\n", total_saved_blocks);
                fprintf(kv_fp, "\n");
                for (size_t i = 0; i < batch_size; i++) {
                    int64_t req_id = req_ids[i];
                    int32_t seq_len = sl_data[i];
                    fprintf(kv_fp, "req%ld: seq_len=%d, blocks=[", req_id, seq_len);
                    for (size_t b = 0; b < bt_cols; b++) {
                        int bid = bt_data[i * bt_cols + b];
                        if (bid < 0) break;
                        fprintf(kv_fp, "%d%s", bid, (b + 1 < bt_cols && bt_data[i * bt_cols + b + 1] >= 0) ? "," : "");
                    }
                    fprintf(kv_fp, "]\n");
                }
                fclose(kv_fp);
            }
            
            ((ROCmDevice*)device)->syncAndCheck();
            
            printf("[PA_DUMP] R%d L%d: Saved %d blocks for %d requests\n", 
                   round, layer_id, total_saved_blocks, new_requests_count);
        }
        
        auto q_host = device->clone({q_tmp, AllocationType::HOST});
        ((ROCmDevice*)device)->syncAndCheck();
        size_t q_per_req = q_host->size() / batch_size;
        
        int saved_count = 0;
        
        for (size_t i = 0; i < batch_size; i++) {
            int64_t req_id = req_ids[i];
            
            int step;
            {
                std::lock_guard<std::mutex> lock(pa_step_mutex);
                step = ++pa_step_counter[req_id];
            }
            
            if (step != 1 && step != 10 && step != 25) continue;
            
            std::string dir = "/mnt/raid0/yilin/rtp-llm/zztmp/pa_inputs/round" + std::to_string(round) + 
                              "/req" + std::to_string(req_id) + "_step" + std::to_string(step);
            if (!fs::exists(dir)) {
                fs::create_directories(dir);
            }
            
            std::vector<size_t> q_shape;
            for (size_t d = 1; d < q_host->shape().size(); d++) {
                q_shape.push_back(q_host->shape()[d]);
            }
            auto q_slice = device->allocateBuffer({q_host->type(), q_shape, AllocationType::HOST}, {"q_slice"});
            std::memcpy(q_slice->data(), 
                       static_cast<const char*>(q_host->data()) + i * q_per_req * q_host->typeSize(),
                       q_per_req * q_host->typeSize());
            saveBufferDataToTorch(*q_slice, nullptr, dir + "/query.pt");
            
            int32_t seq_len = sl_data[i];
            
            FILE* fp = fopen((dir + "/pa_info.txt").c_str(), "w");
            if (fp) {
                fprintf(fp, "request_id: %ld\n", req_id);
                fprintf(fp, "round: %d\n", round);
                fprintf(fp, "decode_step: %d\n", step);
                fprintf(fp, "batch_idx: %zu\n", i);
                fprintf(fp, "sequence_length: %d (context_lens = %d)\n", seq_len, seq_len + 1);
                fprintf(fp, "block_ids: [");
                for (int b = 0; b < (int)bt_cols; b++) {
                    int bid = bt_data[i * bt_cols + b];
                    if (bid < 0) break;
                    fprintf(fp, "%d%s", bid, (b + 1 < (int)bt_cols && bt_data[i * bt_cols + b + 1] >= 0) ? ", " : "");
                }
                fprintf(fp, "]\n");
                fclose(fp);
            }
            saved_count++;
        }
        
        if (saved_count > 0) {
            printf("[PA_DUMP] R%d: Saved %d requests (step 1/10/25) query to pa_inputs/\n", round, saved_count);
        }
    }
    
    if (key_cache.dtype() == at::kFloat8_e4m3fnuz) {
        K_QScale = Buffer2torchTensor(params.common.kv_cache->kv_scale_buffer, false);
        V_QScale = K_QScale;
        pa_fwd(query,
               key_cache,
               value_cache,
               block_tables,
               context_lens,
               max_num_blocks,
               max_seq_len,
               K_QScale,
               V_QScale,
               out_opt,
               std::nullopt,
               0);
    } else {
        pa_fwd(query,
               key_cache,
               value_cache,
               block_tables,
               context_lens,
               max_num_blocks,
               max_seq_len,
               K_QScale,
               V_QScale,
               out_opt);
    }
}

void runAiterPA(const AttentionModuleParams& params, rtp_llm::DeviceBase* device, Buffer& q_tmp) {
    auto out   = Buffer2torchTensor(params.output, false);
    auto query = Buffer2torchTensor(q_tmp, false);

    if (q_tmp.shape().size() < 3) {
        throw std::runtime_error("aiter_paged_attention only support 3-dim input");
    } else if (q_tmp.shape().size() > 3) {
        query = query.reshape({query.size(0), query.size(1), -1});
    }

    size_t  num_seqs       = q_tmp.shape()[0];
    size_t  num_heads      = params.configs.head_num;
    size_t  head_size      = params.configs.size_per_head;
    int64_t partition_size = 512;
    int64_t max_seq_len =
        device->nativeGraphCapturing() ? device->initParams().max_seq_len : params.common.decoder_max_seq_len + 1;
    size_t    max_num_partitions = (max_seq_len + partition_size - 1) / partition_size;
    auto      datatype           = params.output.type();
    BufferPtr exp_sums_buffer    = device->allocateBuffer(
        {rtp_llm::DataType::TYPE_FP32, {num_seqs, num_heads, max_num_partitions}, AllocationType::DEVICE},
        {"exp_sums"});
    auto exp_sums = Buffer2torchTensor(exp_sums_buffer, false);

    BufferPtr max_logits_buffer = device->allocateBuffer(
        {rtp_llm::DataType::TYPE_FP32, {num_seqs, num_heads, max_num_partitions}, AllocationType::DEVICE},
        {"max_logits"});
    auto max_logits = Buffer2torchTensor(max_logits_buffer, false);

    BufferPtr tmp_out_buffer = device->allocateBuffer(
        {datatype, {num_seqs, num_heads, max_num_partitions, head_size}, AllocationType::DEVICE}, {"tmp_out"});
    auto tmp_out = Buffer2torchTensor(tmp_out_buffer, false);

    auto key_cache   = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 0);
    auto value_cache = Buffer2torchTensor(params.common.kv_cache->kv_cache_buffer, false).select(1, 1);
    /*size_t v_cache_offset = params.common.kv_cache->kv_cache_buffer->sizeBytes();
    auto value_cache = Buffer2torchTensorCustom(*params.common.kv_cache->kv_cache_buffer,
                                               {(int64_t)params.common.kv_cache->kv_cache_buffer->shape()[0],
                                                (int64_t)params.common.kv_cache->kv_cache_buffer->shape()[1],
                                                (int64_t)params.common.kv_cache->kv_cache_buffer->shape()[2]},
                                               v_cache_offset);*/

    int64_t num_kv_heads = params.configs.kv_head_num;
    int64_t grp_size     = num_heads / num_kv_heads;
    double  scale        = params.configs.softmax_extra_scale / sqrtf(params.configs.size_per_head * 1.0f);

    int64_t block_size = params.configs.tokens_per_block;

    std::string kv_cache_dtype = key_cache.dtype() == at::kFloat8_e4m3fnuz ? "fp8" : "auto";

    double k_scale = 1.0;
    double v_scale = 1.0;

    std::optional<torch::Tensor> fp8_out_scale;
    std::optional<torch::Tensor> alibi_slopes;

    auto block_tables = Buffer2torchTensor(params.common.kv_cache->kv_cache_block_id, false);
    // int64_t max_num_blocks_per_seq = (int64_t)params.common.kv_cache->kv_cache_block_id->shape()[1];
    // auto block_tables = Buffer2torchTensorCustom(*params.common.kv_cache->kv_cache_block_id,
    //                                             {(int64_t)params.common.kv_cache->kv_cache_block_id->shape()[0],
    //                                              max_num_blocks_per_seq,
    //                                             }, 0);

    auto aiter_attn = (AiterAttnParams*)params.common.decode_aiter_attn.get();
    if (!aiter_attn) {
        throw std::runtime_error("aiter_attn must be setting when using aiter pa");
    }

    auto context_lens = aiter_attn->sequence_lengths_t;
    if (max_seq_len <= 16384) {
        int64_t x        = 16 / key_cache.element_size();
        auto    kv_sizes = value_cache.sizes();
        out              = out.view({int64_t(num_seqs), int64_t(num_heads), int64_t(head_size)});
        exp_sums   = exp_sums.view({int64_t(num_seqs), int64_t(num_kv_heads), int64_t(max_num_partitions), grp_size});
        max_logits = max_logits.view({int64_t(num_seqs), int64_t(num_kv_heads), int64_t(max_num_partitions), grp_size});
        tmp_out    = tmp_out.view(
            {int64_t(num_seqs), int64_t(num_kv_heads), int64_t(max_num_partitions), grp_size, int64_t(head_size)});
        query       = query.view({int64_t(num_seqs), int64_t(num_heads), int64_t(head_size)});
        key_cache   = key_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3] / x, kv_sizes[2], x});
        value_cache = value_cache.view({kv_sizes[0], kv_sizes[1], kv_sizes[3], kv_sizes[2]});
        paged_attention_atrex(out,
                              exp_sums,
                              max_logits,
                              tmp_out,
                              query,
                              key_cache,
                              value_cache,
                              context_lens,
                              block_tables,
                              scale,
                              max_seq_len,
                              alibi_slopes);
    } else {
        partition_size = 256;
        max_seq_len =
            device->nativeGraphCapturing() ? device->initParams().max_seq_len : params.common.decoder_max_seq_len + 1;
        max_num_partitions = (max_seq_len + partition_size - 1) / partition_size;
        datatype           = params.output.type();
        exp_sums_buffer    = device->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {num_seqs, num_heads, max_num_partitions}, AllocationType::DEVICE},
            {"exp_sums"});
        exp_sums = Buffer2torchTensor(exp_sums_buffer, false);

        max_logits_buffer = device->allocateBuffer(
            {rtp_llm::DataType::TYPE_FP32, {num_seqs, num_heads, max_num_partitions}, AllocationType::DEVICE},
            {"max_logits"});
        max_logits = Buffer2torchTensor(max_logits_buffer, false);

        tmp_out_buffer = device->allocateBuffer(
            {datatype, {num_seqs, num_heads, max_num_partitions, head_size}, AllocationType::DEVICE}, {"tmp_out"});
        tmp_out = Buffer2torchTensor(tmp_out_buffer, false);
        paged_attention(out,
                        exp_sums,
                        max_logits,
                        tmp_out,
                        query,
                        key_cache,
                        value_cache,
                        num_kv_heads,
                        scale,
                        block_tables,
                        context_lens,
                        block_size,
                        max_seq_len,
                        alibi_slopes,
                        kv_cache_dtype,
                        k_scale,
                        v_scale,
                        fp8_out_scale,
                        partition_size);
    }
    return;
}

}  // namespace rtp_llm

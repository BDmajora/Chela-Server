#include "gpu-kernels.h"
#include "explicit-fixture-inputs.h"
// Exercise the real CPU sink, io_uring, borrowed H2D and explicit MXFP4 GPU
// format at synthetic geometry. Cache, upload and failure assertions use the
// production sink and tier without a full model dependency.
#define main original_sink_test_main
#include "test-kimi-cpu-sink.cpp"
#undef main

int main() {
    fixture_inputs inputs; inputs.set_runtime_services(test_gpu_services()); try {
    ggml_cpu_init();
    inputs.set("ready_first", "1");
    inputs.set("cache_policy", "lru");
    inputs.set("cache_allocation", "uniform");
    inputs.set("cache_keep", "0");
    inputs.set("tier.policy", "lru");
    inputs.set("tier.min_count", "0");
    fixture f;f.latent=256;f.hidden=256;f.experts=4;f.type=f.down_type=GGML_TYPE_MXFP4;make_synthetic(f);
    const auto ram_bytes=2*triplet_bytes(f);
    const auto gpu_bytes=std::to_string(2*triplet_bytes(f)); // one GPU slot/layer
    struct options { bool exclusive,async,hidden; };
    size_t compared=0,checks=0;
    for(const auto cfg : {options{false,false,false},options{true,false,false},options{true,true,false},options{true,true,true},options{false,true,true}}) {
        inputs.set("vram_budget", "0");
        inputs.set("vram_exclusive", "0");
        auto cpu=make_sink(inputs, f,16,2);
        inputs.set("vram_budget", gpu_bytes.c_str());
        inputs.set("vram_exclusive", cfg.exclusive?"1":"0");
        inputs.set("vram_async", cfg.async?"1":"0");
        inputs.set("tier.mxfp4_gpu_hidden", cfg.hidden?"1":"0");
        auto mixed=make_sink(inputs, f,16,2);
        configure_cache(cpu.get(),ram_bytes,{1,2},1);configure_cache(mixed.get(),ram_bytes,{1,2},1);
        for(int token=1;token<=10;++token) {
            const bool decode=token!=7;
            const uint64_t epoch=token<9?17:18;
            begin_cache(cpu.get(),epoch,token,decode);begin_cache(mixed.get(),epoch,token,decode);
            for(int layer:{1,2}) {
                const std::vector<int32_t> ids=token==3?std::vector<int32_t>{2,1,3}:
                    token==4?std::vector<int32_t>{2,1}:token==5?std::vector<int32_t>{2,2,0}:std::vector<int32_t>{0,1,2};
                std::vector<float> weights(ids.size(),.2f);weights[0]=.6f;
                std::vector<float> input(f.latent);
                for(int i=0;i<f.latent;++i)input[i]=.031f*std::sin(float(i*7+token*31))+.1f;
                std::vector<float> a(f.latent),b(f.latent);tensors meta(f,layer);char error[512]={};
                kimi_cpu_sink_stats stats={};
                for(int arm=0;arm<2;++arm){auto sink=arm?mixed.get():cpu.get();auto & out=arm?b:a;
                    require(kimi_cpu_sink_run(sink,meta.role[0],meta.role[1],meta.role[2],input.data(),ids.data(),weights.data(),ids.size(),4,25,out.data(),&stats,error,sizeof(error)),error);
                    require(stats.read_errors==0 && stats.pending_reads==0 && stats.fallback_count==0,"hybrid work did not drain");++checks;
                }
                require(!std::memcmp(a.data(),b.data(),a.size()*sizeof(float)),"exclusive/async/hidden chain changed output");compared+=a.size();
                const auto state=cache_snapshot(mixed.get());
                if(token==1 && cfg.exclusive) { require(state.valid_bytes==0,"new GPU copy still occupied RAM");++checks; }
            }
        }
        // Resolve a later request beyond EOF after other cache reads were
        // submitted. Any queued uploads must drain before RAM is released.
        begin_cache(mixed.get(),18,11,false);begin_cache(mixed.get(),18,12);
        f.fault=fixture::past_eof;f.fault_expert=3;f.fault_role=2;
        tensors meta(f,1);std::vector<int32_t> ids={0,1,3};std::vector<float> weights(3,.3f),input(f.latent,.1f),out(f.latent);
        char error[512]={};kimi_cpu_sink_stats stats={};
        require(!kimi_cpu_sink_run(mixed.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),ids.data(),weights.data(),3,4,25,out.data(),&stats,error,sizeof(error)),"I/O failure accepted");
        require(stats.pending_reads==0 && stats.fallback_count==0,"I/O failure returned with live work");++checks;
        f.fault=fixture::none;f.fault_expert=f.fault_role=-1;
        begin_cache(mixed.get(),18,13,false);begin_cache(mixed.get(),18,14);
        require(kimi_cpu_sink_run(mixed.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),ids.data(),weights.data(),3,4,25,out.data(),&stats,error,sizeof(error)),error);++checks;
        std::printf("EXCLUSIVE_MODE_PASS exclusive=%d async=%d gpu_hidden=%d\n",int(cfg.exclusive),int(cfg.async),int(cfg.hidden));
    }
    // With free slots in its layer RAM keeps the copy of what went up: an eviction from the tier finds it there.
    {
        inputs.set("vram_budget", gpu_bytes.c_str());
        inputs.set("vram_exclusive", "1");
        inputs.set("vram_async", "0");
        auto roomy=make_sink(inputs, f,16,2);
        configure_cache(roomy.get(),2*ram_bytes,{1,2},2);
        tensors meta(f,1);const std::vector<int32_t> ids={0};const std::vector<float> weights={1.f};
        std::vector<float> input(f.latent,.1f),out(f.latent);
        for(int token=1;token<=2;++token){
            begin_cache(roomy.get(),19,token);
            char error[512]={};kimi_cpu_sink_stats stats={};
            const bool ok=kimi_cpu_sink_run(roomy.get(),meta.role[0],meta.role[1],meta.role[2],input.data(),ids.data(),weights.data(),1,4,25,out.data(),&stats,error,sizeof(error));
            require(ok,std::string("roomy run: ")+error);
            require(cache_snapshot(roomy.get()).valid_bytes==triplet_bytes(f),"a layer with free slots gave up its copy");++checks;
        }
    }
    std::printf("REAL_GPU_EXCLUSIVE_PASS compared=%zu assertions=%zu\n",compared,checks);return 0;
}catch(const std::exception & e){std::fprintf(stderr,"REAL_GPU_EXCLUSIVE_FAIL: %s\n",e.what());return 1;} }

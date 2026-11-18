#include <cstdio>
#include <vector>
#include <fstream>
#include "inkling_moe_base_model.h"
#include "lsxcommon/core.h"
#include "lsxcommon/model_factory.h"
#include "lsxcommon/code_selfcheck.h"
int main(int argc,char**argv){
  lsxcommon::Core::Initialize(); lsxcommon::EnsureModelRegistry();
  lsxcommon::ModelInitConfig cfg; cfg.model_path=argv[1]; cfg.model_type="inkling";
  cfg.fast_math_mode=lsxcommon::FastMathMode::kOff;
  auto m=lsxcommon::LsxModelFactory::Create(cfg);
  if(!m->Load()){std::fprintf(stderr,"load failed\n");return 1;}
  std::ifstream f("tests/fixtures/code/inkling.json");
  std::string js((std::istreambuf_iterator<char>(f)),{});
  lsxcommon::code::CodeGolden g; std::string err;
  if(!lsxcommon::code::ParseCodeGolden(js,g,&err)){std::fprintf(stderr,"parse: %s\n",err.c_str());return 1;}
  auto &c=g.cases[2];
  lsxcommon::ModelRequest req; req.input_ids=c.input_ids; req.max_gen_tokens=c.max_new_tokens; req.ignore_eos=c.ignore_eos;
  auto r=m->Infer(req);
  std::printf("CODE-03 gen %zu tokens:\n",r.gen_tokens.size());
  for(size_t i=0;i<r.gen_tokens.size();++i){
    int32_t id=r.gen_tokens[i]; char buf[64]; size_t cap=sizeof(buf);
    lsxcommon::arrow_tokenizer_decode(m->Tokenizer(),&id,1,buf,&cap); buf[cap]=0;
    if(id>=200000 || i<3 || i+3>=r.gen_tokens.size())
      std::printf("[%zu] %d '%s'\n",i,id,buf);
  }
  m->Unload(); lsxcommon::Core::Shutdown(); return 0;
}

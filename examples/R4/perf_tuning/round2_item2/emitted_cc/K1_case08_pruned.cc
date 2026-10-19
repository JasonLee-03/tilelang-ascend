#include "tl_templates/ascend/common.h"
#include "acl/acl.h"
#include <runtime/rt_ffts.h>
using namespace Catlass;
using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

extern "C" __global__ __aicore__ void main_kernel( GM_ADDR logits_handle,  GM_ADDR top_ks_handle,  GM_ADDR top_ps_handle,  GM_ADDR min_ps_handle,  GM_ADDR m_out_handle,  GM_ADDR s_out_handle,  GM_ADDR tau_out_handle,  GM_ADDR valid_handle, uint64_t fftsAddr) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  AscendC::TPipe pipe;

  AscendC::GlobalTensor<float> logits;
  logits.SetGlobalBuffer((__gm__ float*)logits_handle);
  AscendC::GlobalTensor<int> top_ks;
  top_ks.SetGlobalBuffer((__gm__ int*)top_ks_handle);
  AscendC::GlobalTensor<float> top_ps;
  top_ps.SetGlobalBuffer((__gm__ float*)top_ps_handle);
  AscendC::GlobalTensor<float> min_ps;
  min_ps.SetGlobalBuffer((__gm__ float*)min_ps_handle);
  AscendC::GlobalTensor<float> m_out;
  m_out.SetGlobalBuffer((__gm__ float*)m_out_handle);
  AscendC::GlobalTensor<float> s_out;
  s_out.SetGlobalBuffer((__gm__ float*)s_out_handle);
  AscendC::GlobalTensor<float> tau_out;
  tau_out.SetGlobalBuffer((__gm__ float*)tau_out_handle);
  AscendC::GlobalTensor<int> valid;
  valid.SetGlobalBuffer((__gm__ int*)valid_handle);

  AscendC::TBuf<AscendC::TPosition::A2> ascend_l0a;
  pipe.InitBuffer(ascend_l0a, 65536);
  AscendC::TBuf<AscendC::TPosition::B2> ascend_l0b;
  pipe.InitBuffer(ascend_l0b, 65536);
  AscendC::TBuf<AscendC::TPosition::A1> ascend_l1; pipe.InitBuffer(ascend_l1, 524032);
  AscendC::TBuf<AscendC::TPosition::CO1> ascend_l0c; pipe.InitBuffer(ascend_l0c, 131072);
  AscendC::TBuf<AscendC::TPosition::VECCALC> ascend_ub; pipe.InitBuffer(ascend_ub, 196352);
  pipe.Destroy();
  auto cid = AscendC::GetBlockIdx();
  if ASCEND_IS_AIV {
    cid = cid / 2;
  }
  auto tau_k = ascend_ub.GetWithOffset<float>(1, 0);
  auto tau_p = ascend_ub.GetWithOffset<float>(1, 32);
  auto prefix = ascend_ub.GetWithOffset<float>(1, 64);
  auto tau_u = ascend_ub.GetWithOffset<float>(1, 96);
  auto okp = ascend_ub.GetWithOffset<int>(1, 128);
  auto tieok = ascend_ub.GetWithOffset<int>(1, 160);
  auto vflag = ascend_ub.GetWithOffset<int>(1, 192);
  auto tie_m = ascend_ub.GetWithOffset<float>(128, 224);
  auto tie_one = ascend_ub.GetWithOffset<float>(128, 736);
  auto tau_run = ascend_ub.GetWithOffset<float>(1, 1248);
  auto k_i = ascend_ub.GetWithOffset<int>(1, 1280);
  auto p_i = ascend_ub.GetWithOffset<float>(1, 1312);
  auto mu_i = ascend_ub.GetWithOffset<float>(1, 1344);
  auto m_run = ascend_ub.GetWithOffset<float>(1, 1376);
  auto s_run = ascend_ub.GetWithOffset<float>(1, 1408);
  auto run_v = ascend_ub.GetWithOffset<float>(128, 1440);
  auto tmp_ub = ascend_ub.GetWithOffset<uint8_t>(32768, 10176);
  auto z_ub = ascend_ub.GetWithOffset<float>(2048, 1952);
  auto m_blk = ascend_ub.GetWithOffset<float>(1, 10144);
  auto m_old = ascend_ub.GetWithOffset<float>(1, 42944);
  auto tmp = ascend_ub.GetWithOffset<float>(1, 42976);
  auto m2d = ascend_ub.GetWithOffset<float>(2048, 43008);
  auto e_ub = ascend_ub.GetWithOffset<float>(2048, 51200);
  auto s_blk = ascend_ub.GetWithOffset<float>(1, 59392);
  auto l_dst = ascend_ub.GetWithOffset<float>(256, 59424);
  auto l_val = ascend_ub.GetWithOffset<float>(128, 60448);
  auto merged = ascend_ub.GetWithOffset<float>(256, 60960);
  auto m_dst = ascend_ub.GetWithOffset<float>(256, 61984);
  auto rv2 = ascend_ub.GetWithOffset<float>(128, 63008);
  auto mk2 = ascend_ub.GetWithOffset<float>(128, 63520);
  auto q2 = ascend_ub.GetWithOffset<float>(128, 64032);
  auto sk2 = ascend_ub.GetWithOffset<float>(128, 64544);
  auto tie_tkbc = ascend_ub.GetWithOffset<float>(128, 65056);
  auto tie_mask = ascend_ub.GetWithOffset<uint8_t>(16, 65568);
  auto n_gt = ascend_ub.GetWithOffset<float>(1, 65600);
  auto n_eq = ascend_ub.GetWithOffset<float>(1, 65632);
  auto tau = ascend_ub.GetWithOffset<float>(1, 65664);
  auto vid = AscendC::GetSubBlockIdx();
  if ASCEND_IS_AIV {
    tl::ascend::Fill<float>(tau_k[0], 0.000000e+00f, 1);
    tl::ascend::Fill<float>(tau_p[0], 0.000000e+00f, 1);
    tl::ascend::Fill<float>(prefix[0], 0.000000e+00f, 1);
    tl::ascend::Fill<float>(tau_u[0], 0.000000e+00f, 1);
    tl::ascend::Fill<int>(okp[0], 0, 1);
    tl::ascend::Fill<int>(tieok[0], 0, 1);
    tl::ascend::Fill<int>(vflag[0], 0, 1);
    tl::ascend::Fill<float>(tie_m[0], 0.000000e+00f, 128);
    tl::ascend::Fill<float>(tie_one[0], 1.000000e+00f, 128);
    tl::ascend::Fill<float>(tau_run[0], -CUDART_INF_F, 1);
    tl::ascend::copy_gm_to_ub<int, 1>(k_i[0], top_ks[((cid * 2) + vid)], 4, 1, 1, 0);
    tl::ascend::copy_gm_to_ub<float, 1, 1>(p_i[0], top_ps[((cid * 2) + vid)], 4, 1, 1, 0.000000e+00f);
    tl::ascend::copy_gm_to_ub<float, 1, 1>(mu_i[0], min_ps[((cid * 2) + vid)], 4, 1, 1, 0.000000e+00f);
    tl::ascend::Fill<float>(m_run[0], -CUDART_INF_F, 1);
    tl::ascend::Fill<float>(s_run[0], 0.000000e+00f, 1);
    tl::ascend::Fill<float>(run_v[0], -CUDART_INF_F, 128);
    for (int32_t b = 0; b < 8; ++b) {
      tl::ascend::copy_gm_to_ub<float, 2048>(z_ub[0], logits[(((cid * 32768) + (vid * 16384)) + (b * 2048))], 65536, 1, 2048, 0.000000e+00f);
      AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(1);
      AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(1);
      tl::ascend::reduce_max<float, 1, 2048, -1>(m_blk[0], z_ub[0], tmp_ub[0], true);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Max(m_old[0], m_run[0], m_blk[0], 1);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(tmp[0], m_run[0], m_old[0], 1);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(tmp[0], tmp[0], 1);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Mul(s_run[0], s_run[0], tmp[0], 1);
      AscendC::Max(m_run[0], m_run[0], m_blk[0], 1);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::Broadcast<float, 2, 1, false>(m2d[0],m_run[0],(uint32_t[]){1, 2048}, (uint32_t[]){1, 1});
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Sub(e_ub[0], z_ub[0], m2d[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Exp(e_ub[0], e_ub[0], 2048);
      AscendC::PipeBarrier<PIPE_V>();
      tl::ascend::reduce_sum<float,  1,  2048,  -1>(s_blk[0], e_ub[0], tmp_ub[0], true);
      AscendC::PipeBarrier<PIPE_V>();
      AscendC::Add(s_run[0], s_run[0], s_blk[0], 1);
      AscendC::PipeBarrier<PIPE_ALL>();
      if (tau_run.GetValue(0) <= m_blk.GetValue(0)) {
        tl::ascend::TopK<float>(l_dst[0], z_ub[0], tmp_ub.ReinterpretCast<float>()[0], 128, 64, 2048);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::GatherMask<float>(l_val[0], l_dst[0], 1);
        tl::ascend::copy_ub_to_ub<float, float, 128>(merged[0], run_v[0], 1, 128, 128, 1, 128, 256);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::copy_ub_to_ub<float, float, 128>(merged[128], l_val[0], 1, 128, 128, 1, 128, 256);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::TopK<float>(m_dst[0], merged[0], tmp_ub.ReinterpretCast<float>()[0], 128, 8, 256);
        AscendC::PipeBarrier<PIPE_V>();
        tl::ascend::GatherMask<float>(run_v[0], m_dst[0], 1);
        AscendC::SetFlag<AscendC::HardEvent::V_S>(2);
        AscendC::WaitFlag<AscendC::HardEvent::V_S>(2);
        tau_run.SetValue(0, run_v.GetValue(127));
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    tl::ascend::copy_ub_to_ub<float, float, 128>(rv2[0], run_v[0], 1, 128, 128, 1, 128, 128);
    tl::ascend::Broadcast<float, 2, 1, false>(mk2[0],m_run[0],(uint32_t[]){1, 128}, (uint32_t[]){1, 1});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Sub(q2[0], rv2[0], mk2[0], 128);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Exp(q2[0], q2[0], 128);
    tl::ascend::Broadcast<float, 2, 1, false>(sk2[0],s_run[0],(uint32_t[]){1, 128}, (uint32_t[]){1, 1});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Div(q2[0], q2[0], sk2[0], 128);
    tl::ascend::Fill<float>(tau_k[0], -CUDART_INF_F, 1);
    AscendC::PipeBarrier<PIPE_ALL>();
    if ((1 <= k_i.GetValue(0)) && (k_i.GetValue(0) <= 128)) {
      tau_k.SetValue(0, run_v.GetValue((k_i.GetValue(0) - 1)));
    }
    AscendC::PipeBarrier<PIPE_ALL>();
    tl::ascend::Fill<float>(tau_u[0], -CUDART_INF_F, 1);
    tl::ascend::Fill<float>(tau_p[0], -CUDART_INF_F, 1);
    tl::ascend::Fill<float>(prefix[0], 0.000000e+00f, 1);
    for (int32_t l = 0; l < 128; ++l) {
      AscendC::PipeBarrier<PIPE_ALL>();
      if (prefix.GetValue(0) <= p_i.GetValue(0)) {
        tau_p.SetValue(0, run_v.GetValue(l));
        prefix.SetValue(0, (prefix.GetValue(0) + q2.GetValue(l)));
      }
      AscendC::PipeBarrier<PIPE_ALL>();
      AscendC::PipeBarrier<PIPE_ALL>();
    }
    int32_t condval;
    if ((1.000000e+00f <= p_i.GetValue(0))) {
      condval = 1;
    } else {
      int32_t condval_1;
      if ((p_i.GetValue(0) < prefix.GetValue(0))) {
        condval_1 = 1;
      } else {
        condval_1 = 0;
      }
      condval = condval_1;
    }
    okp.SetValue(0, condval);
    tl::ascend::Broadcast<float, 1, 0, false>(tie_tkbc[0],tau_k[0],(uint32_t[]){128}, (uint32_t[]){1});
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Compare(tie_mask[0], run_v[0], tie_tkbc[0], AscendC::CMPMODE::GT, 128);
    AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(tie_m[0], tie_mask[0], tie_one[0], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 128);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::reduce_sum<float,  1,  128,  -1>(n_gt[0], tie_m[0], tmp_ub[0], true);
    AscendC::Compare(tie_mask[0], run_v[0], tie_tkbc[0], AscendC::CMPMODE::EQ, 128);
    AscendC::PipeBarrier<PIPE_V>();
AscendC::Select<float, uint8_t>(tie_m[0], tie_mask[0], tie_one[0], static_cast<float>(0.000000e+00f), AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, 128);
    AscendC::PipeBarrier<PIPE_V>();
    tl::ascend::reduce_sum<float,  1,  128,  -1>(n_eq[0], tie_m[0], tmp_ub[0], true);
    AscendC::Max(tau[0], tau_k[0], tau_p[0], 1);
    AscendC::PipeBarrier<PIPE_V>();
    AscendC::Max(tau[0], tau[0], tau_u[0], 1);
    AscendC::SetFlag<AscendC::HardEvent::V_S>(5);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(5);
    int32_t condval_2;
    if ((tau_k.GetValue(0) < tau.GetValue(0))) {
      condval_2 = 1;
    } else {
      int32_t condval_3;
      if ((((float)k_i.GetValue(0)) < (n_gt.GetValue(0) + n_eq.GetValue(0)))) {
        condval_3 = 0;
      } else {
        int32_t condval_4;
        if (((n_gt.GetValue(0) + n_eq.GetValue(0)) < 1.280000e+02f)) {
          condval_4 = 1;
        } else {
          int32_t condval_5;
          if ((n_eq.GetValue(0) <= 1.000000e+00f)) {
            condval_5 = 1;
          } else {
            condval_5 = 0;
          }
          condval_4 = condval_5;
        }
        condval_3 = condval_4;
      }
      condval_2 = condval_3;
    }
    tieok.SetValue(0, condval_2);
    int32_t condval_6;
    if (((((1 <= k_i.GetValue(0)) && (k_i.GetValue(0) <= 128)) && (okp.GetValue(0) == 1)) && (tieok.GetValue(0) == 1))) {
      condval_6 = 1;
    } else {
      condval_6 = 0;
    }
    vflag.SetValue(0, condval_6);
    tl::ascend::copy_ub_to_gm<float, 1, 1>(m_out[((cid * 2) + vid)], m_run[0], 4, 1, 1);
    tl::ascend::copy_ub_to_gm<float, 1, 1>(s_out[((cid * 2) + vid)], s_run[0], 4, 1, 1);
    tl::ascend::copy_ub_to_gm<float, 1>(tau_out[((cid * 2) + vid)], tau[0], 4, 1, 1);
    AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(6);
    AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(6);
    tl::ascend::copy_ub_to_gm<int, 1>(valid[((cid * 2) + vid)], vflag[0], 4, 1, 1);
  }
}

void main_kernel_tiling() {
}

extern "C" void call(uint8_t* logits_handle, uint8_t* top_ks_handle, uint8_t* top_ps_handle, uint8_t* min_ps_handle, uint8_t* m_out_handle, uint8_t* s_out_handle, uint8_t* tau_out_handle, uint8_t* valid_handle, aclrtStream stream) {
  uint32_t fftsLen{0};
  uint64_t fftsAddr{0};
  rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
  main_kernel_tiling();
  main_kernel<<<2, nullptr, stream>>>(logits_handle, top_ks_handle, top_ps_handle, min_ps_handle, m_out_handle, s_out_handle, tau_out_handle, valid_handle, fftsAddr);
}

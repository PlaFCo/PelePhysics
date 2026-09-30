#ifndef REACTORUTILSISOBARIC_H
#define REACTORUTILSISOBARIC_H

#include <utility>
#include <AMReX_Gpu.H>
#include "mechanism.H"
#include "PelePhysics.H"
#include "ReactorTypes.H"
#include "AMReX_Sundials.H"
#include "ReactorUtils.H"

#include <cvode/cvode.h>

#include <nvector/nvector_serial.h>

#ifdef AMREX_USE_HIP
#include <nvector/nvector_hip.h>
#endif

#ifdef AMREX_USE_SYCL
#include <nvector/nvector_sycl.h>
#endif

#ifdef AMREX_USE_CUDA
#include <nvector/nvector_cuda.h>
#endif

namespace pele::physics::reactions::utils {

// Overload of box_flatten for CVODE Isobaric reactor
template <typename OrderType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
box_flatten(
    IsobaricTag /*tag*/,
    const int icell,
    const int i,
    const int j,
    const int k,
    const int ncells,
    const bool clean_init_massfrac,
    amrex::Array4<const amrex::Real> const& rhoY,
    amrex::Array4<const amrex::Real> const& frcExt,
    amrex::Array4<const amrex::Real> const& temperature,
    amrex::Array4<const amrex::Real> const& frcEExt,
    amrex::Real* y_vect,
    amrex::Real* src_vect,
    amrex::Real* p_vect,
    amrex::Real* hsrc_vect)
{
#ifndef USE_MANIFOLD_EOS
    amrex::Real Y[NUM_SPECIES];
    amrex::Real rho = 0.0;
    amrex::Real rho_clean = 0.0; // for normalizing Y

    for (int n = 0; n < NUM_SPECIES; n++)
    {
        const amrex::Real rhoYn = rhoY(i, j, k, n);
        rho += rhoYn;
        Y[n] = clean_init_massfrac ? amrex::max<amrex::Real>(0.0, rhoYn) : rhoYn;
        rho_clean += Y[n];
        src_vect[spec_index<OrderType>(n, icell, ncells)] = frcExt(i, j, k, n);
    }
    const amrex::Real rho_clean_inv = 1.0/rho_clean;
    for (amrex::Real& Yn : Y)
    {
        Yn *= rho_clean_inv;
    }
    const amrex::Real T = temperature(i, j, k, 0);
    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real P0 = 0.0;
    eos.RTY2P(rho, T, Y, P0);
    p_vect[icell] = P0;
    hsrc_vect[icell] = frcEExt(i, j, k, 0) / rho;

    for (int n = 0; n < NUM_SPECIES; n++) 
    {
        y_vect[vec_index<OrderType>(n, icell, ncells)] = Y[n];
    }
    y_vect[vec_index<OrderType>(NUM_SPECIES, icell, ncells)] = T;
#else
    amrex::Abort("Isobaric reactor is not available with the Manifold EOS");
#endif
}

// Overload of box_unflatten
template <typename OrderType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
box_unflatten(
    IsobaricTag /*tag*/,
    const int icell,
    const int i,
    const int j,
    const int k,
    const int ncells,
    amrex::Array4<amrex::Real> const& rhoY,
    amrex::Array4<amrex::Real> const& temperature,
    amrex::Array4<amrex::Real> const& rhoE,
    amrex::Array4<amrex::Real> const& FC_in,
    const amrex::Real* y_vect,
    const amrex::Real* p_vect,
    long int const& FCunt)
{
#ifndef USE_MANIFOLD_EOS
    amrex::Real Y[NUM_SPECIES];
    amrex::Real Ysum = 0.0;
    for (int n = 0; n < NUM_SPECIES; n++)
    {
        Y[n] = y_vect[vec_index<OrderType>(n, icell, ncells)];
        Ysum += Y[n];
    }
    // Chemistry conserves sum Y = 1, renormalize anyway
    const amrex::Real Ysum_inv = 1.0 / Ysum;
    for (amrex::Real& Yn : Y)
    {
        Yn *= Ysum_inv;
    }
    const amrex::Real T = y_vect[vec_index<OrderType>(NUM_SPECIES, icell, ncells)];

    auto eos = pele::physics::PhysicsType::eos();
    amrex::Real rho = 0.0;
    eos.PYT2R(p_vect[icell], Y, T, rho);
    for (int n = 0; n < NUM_SPECIES; n++)
    {
        rhoY(i, j, k, n) = rho * Y[n];
    }
    temperature(i, j, k, 0) = T;

    amrex::Real h = 0.0;
    eos.RTY2H(rho, T, Y, h);
    rhoE(i, j, k, 0) = rho * h;

    FC_in(i, j, k, 0) = static_cast<amrex::Real>(FCunt);
#else
    amrex::Abort("Isobaric reactor not available with the Manifold EOS");
#endif
}

// Overload of the RHS kernel
template <typename OrderType>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
fKernelSpec(
    IsobaricTag /*tag*/,
    const int icell,
    const int ncells,
    const amrex::Real* yvec_d,
    amrex::Real* ydot_d,
    const amrex::Real* p_init,
    const amrex::Real* hsrc_ext,
    const amrex::Real* rYs,
    const pele::physics::eos::EosParm<pele::physics::PhysicsType::eos_type>*
        eosparm = nullptr)
{
#ifndef USE_MANIFOLD_EOS
    auto eos = pele::physics::PhysicsType::eos(eosparm);

    amrex::Real Y[NUM_SPECIES];
    for (int n = 0; n < NUM_SPECIES; n++)
    {
        Y[n] = yvec_d[vec_index<OrderType>(n, icell, ncells)];
    }
    const amrex::Real T = yvec_d[vec_index<OrderType>(NUM_SPECIES, icell, ncells)];

    amrex::Real rho = 0.0;
    eos.PYT2R(p_init[icell], Y, T, rho);
    const amrex::Real rho_inv = 1.0/rho;

    amrex::Real wdot[NUM_SPECIES] = {0.0};  // production rate
    amrex::Real hi[NUM_SPECIES] = {0.0};    // specific enthalpies
    amrex::Real cp = 0.0;
    eos.RTY2WDOT(rho, T, Y, wdot);
    eos.RTY2Hi(rho, T, Y, hi);
    eos.RTY2Cp(rho, T, Y, cp);

    amrex::Real sum_hf = 0.0;
    for (int n = 0; n < NUM_SPECIES; n++)
    {
        const amrex::Real dYdt = 
            (wdot[n] + rYs[spec_index<OrderType>(n, icell, ncells)]) * rho_inv;
        ydot_d[vec_index<OrderType>(n, icell, ncells)] = dYdt;
        sum_hf += hi[n] * dYdt;
    }
    ydot_d[vec_index<OrderType>(NUM_SPECIES, icell, ncells)] = 
        (hsrc_ext[icell] - sum_hf) / cp;
#else
    amrex::Abort("Isobaric reactor not available with Manifold EOS");
#endif
}
} // namespace
#endif
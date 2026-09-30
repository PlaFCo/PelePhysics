#ifndef REACTORTYPES_H
#define REACTORTYPES_H

#include <AMReX.H>

namespace pele::physics::reactions {

struct IsobaricTag
{
    // Empty struct to help with function overloading
    // To select the isobaric versions when using CP reactor
};

struct ReactorTypes
{
  static constexpr int e_reactor_type = 1;
  static constexpr int h_reactor_type = 2;
  static constexpr int p_reactor_type = 3;

  // allow_isobaric to ensure CP is not used with Arkode/BDF/RK64/Null
  static void check_reactor_type(const int reactor_type, const bool allow_isobaric = false)
  {
    const bool is_eh = (reactor_type == e_reactor_type) ||
                       (reactor_type == h_reactor_type);

    const bool is_p = (reactor_type == p_reactor_type);

    if (is_p && !allow_isobaric)
    {
      amrex::Abort(
        "reactor_type = 3 (const pressure) is only implemented in ReactorCvode"
      );
    }

    if (!is_eh && !is_p)
    {
      amrex::Abort("Wrong reactor type. Choose between 1 (e), 2 (h) or 3(p).");
    }
  }
};
} // namespace pele::physics::reactions
#endif

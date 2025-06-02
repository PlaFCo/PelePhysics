#ifndef REACTORTYPES_H
#define REACTORTYPES_H

#include <AMReX.H>

namespace pele::physics::reactions {

struct ReactorTypes
{
  static constexpr int e_reactor_type = 1;
  static constexpr int h_reactor_type = 2;
#ifdef PELE_USE_NLTE
  static constexpr int hfa_reactor_type = 3;
  static constexpr int hte_reactor_type = 4;
#endif // PELE_USE_NLTE
  static void check_reactor_type(const int reactor_type)
  {
    if (
      (reactor_type != ReactorTypes::e_reactor_type) &&
      (reactor_type != ReactorTypes::h_reactor_type) 
#ifdef PELE_USE_NLTE
      && (reactor_type != ReactorTypes::hfa_reactor_type) &&
      (reactor_type != ReactorTypes::hte_reactor_type)
#endif // PELE_USE_NLTE
    ) 
      {
      amrex::Abort("Wrong reactor type. Choose between 1 (e) or 2 (h) or 3 and 4.");
    }
  }
};
} // namespace pele::physics::reactions
#endif

#ifndef UI_MENU_CUH
#define UI_MENU_CUH

#include "config/config.cuh"

/// ----------------------------------------------------------------------------
/// @file menu.cuh
/// @brief Console menu for building a RuntimeConfig (y/n booleans, defaults on Enter).
/// ----------------------------------------------------------------------------

/// ----------------------------------------------------------------------------
/// @brief Prompt the user for a complete runtime configuration.
/// @details
///  - Booleans use 'y' / 'n' (case-insensitive). Enter keeps the default.
///  - Numeric inputs show the default and a suggested range. Enter keeps the default.
///  - Scene selection is per-scene with y/n, building the final bitmask.
/// ----------------------------------------------------------------------------
RuntimeConfig promptUserForConfig();

#endif // UI_MENU_CUH
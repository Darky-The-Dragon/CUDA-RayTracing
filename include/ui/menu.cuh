/**
* @file menu.cuh
 * @brief Console menu for building a RuntimeConfig.
 * @details
 *  - Booleans use 'y' / 'n' (case-insensitive). Enter keeps the default.
 *  - Numeric inputs show the default and a suggested range. Enter keeps the default.
 *  - Scene selection prompts per scene and builds a bitmask (see SceneBits).
 *  - Host-only utility; CPU builds call this to collect runtime settings.
 */

#pragma once

#include "config/config.cuh"

/**
 * @brief Prompt the user for a complete runtime configuration via stdin/stdout.
 * @details
 *  Prompts for seeds, resolution, scene mask, post-FX options, and debug flags.
 *  Each prompt displays current defaults; pressing Enter accepts the default.
 *  Invalid input is re-asked until a valid value is provided or default accepted.
 *
 * @return Fully populated RuntimeConfig reflecting the user's choices.
 */
RuntimeConfig promptUserForConfig();

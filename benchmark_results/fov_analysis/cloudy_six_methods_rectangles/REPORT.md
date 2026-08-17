# Six registered split-image rectangles

This run uses the full rectangular extent of every VIS image in each `split_images` directory. It does not remove padding or infer irregular valid-content masks.

After content-based registration, each native rectangle is represented with a rectilinear transform containing only independent x/y scale and translation. Rotation, shear and perspective are disabled, so every footprint remains an axis-aligned rectangle in the global canvas.

## Result

- All six registered rectangles contain the final intersection by definition.
- The rectangle that most strongly determines or limits that intersection is `roma_mind_swir_to_vis`.
- RoMa-MIND is the smallest registered rectangle and 97.97% of it is contained in each of the other five registered rectangles.
- The six-way common rectangle occupies 97.97% of RoMa-MIND, leaving only 34,632 pixels outside the common zone.
- The common rectangle occupies 88.44% of RASMD, leaving 218,489 pixels outside it.
- Removing RoMa-MIND increases the common intersection by 195,462 pixels, or 11.70%.
- RASMD is the secondary limiter, with a 34,632 pixel or 2.07% gain when removed.
- The other four methods produce virtually no increase when removed, so they do not materially limit the six-way common zone.

## Common zone

- Intersection bounding box: x=154, y=109, width=1785, height=936.
- Largest axis-aligned rectangle fully contained in all six: x=154, y=109, width=1785, height=936.
- Common intersection area: 1,670,760 pixels in global-canvas coordinates.

Therefore, if one method must be selected as the rectangular reference containing and most closely defining the shared six-method region, it is `roma_mind_swir_to_vis`.

The colors in `limiting_methods_by_region.png` mean area recovered when a method is removed, not alignment quality. Red is larger than cyan because removing RoMa-MIND recovers 195,462 pixels, while removing RASMD recovers 34,632 pixels. For an unambiguous comparison, see `individual_registered_rectangles.png` and `limiting_gain_by_method.png`.

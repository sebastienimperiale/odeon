//! Chart / accent colors: the first slots of the validated reference
//! categorical palette (dataviz skill, July 2026), stepped per surface —
//! the dark column is the same hues re-stepped for the dark surface, not a
//! flip. Scene structure (rods, springs, walls) uses the theme's own neutral
//! strokes instead, so only *meaningful* color comes from here: series k of
//! the observation plot, and the same hue on the observed element in the
//! scene.

use egui::Color32;

const LIGHT: [Color32; 6] = [
    Color32::from_rgb(0x2a, 0x78, 0xd6), // blue
    Color32::from_rgb(0xeb, 0x68, 0x34), // orange
    Color32::from_rgb(0x1b, 0xaf, 0x7a), // aqua
    Color32::from_rgb(0xed, 0xa1, 0x00), // yellow
    Color32::from_rgb(0xe8, 0x7b, 0xa4), // magenta
    Color32::from_rgb(0x00, 0x83, 0x00), // green
];

const DARK: [Color32; 6] = [
    Color32::from_rgb(0x39, 0x87, 0xe5),
    Color32::from_rgb(0xd9, 0x59, 0x26),
    Color32::from_rgb(0x19, 0x9e, 0x70),
    Color32::from_rgb(0xc9, 0x85, 0x00),
    Color32::from_rgb(0xd5, 0x51, 0x81),
    Color32::from_rgb(0x00, 0x83, 0x00),
];

/// Color of observation series `i` (also used to mark the observed element
/// in the scene). Fixed assignment, never cycled: callers with more than
/// four series should rethink the display instead.
pub fn series(i: usize, dark_mode: bool) -> Color32 {
    let table = if dark_mode { &DARK } else { &LIGHT };
    table[i.min(table.len() - 1)]
}
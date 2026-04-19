use glam::Vec3A;
use image::{DynamicImage, GenericImageView, RgbaImage};
use kd_tree::{KdPoint, KdTree};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

const TILE_SIZE: u32 = 32;

fn srgb_to_linear(mut rgb: Vec3A) -> Vec3A {
    for i in 0..3 {
        if rgb[i] <= 0.04045 {
            rgb[i] /= 12.92;
        } else {
            rgb[i] = ((rgb[i] + 0.055) / 1.055).powf(2.4);
        }
    }

    rgb
}

fn rgb_to_xyz(rgb: Vec3A) -> Vec3A {
    let x = 0.4124564 * rgb.x + 0.3575761 * rgb.y + 0.1804375 * rgb.z;
    let y = 0.2126729 * rgb.x + 0.7151522 * rgb.y + 0.0721750 * rgb.z;
    let z = 0.0193339 * rgb.x + 0.119_192 * rgb.y + 0.9503041 * rgb.z;

    Vec3A::new(x, y, z)
}

fn xyz_to_lab(xyz: Vec3A) -> Vec3A {
    fn f(val: f32) -> f32 {
        if val > 0.008856 {
            val.cbrt()
        } else {
            7.787 * val + 16.0 / 116.0
        }
    }

    let xn = 0.95047;
    let yn = 1.0;
    let zn = 1.08883;

    let fx = f(xyz.x / xn);
    let fy = f(xyz.y / yn);
    let fz = f(xyz.z / zn);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);

    Vec3A::new(l, a, b)
}

fn rgb_to_lab(rgb: Vec3A) -> Vec3A {
    xyz_to_lab(rgb_to_xyz(srgb_to_linear(rgb)))
}

struct PaletteEntry {
    image: DynamicImage,
    average_color: Vec3A,
}

impl PaletteEntry {
    pub fn new(image: DynamicImage) -> Self {
        let mut average_color = Vec3A::ZERO;

        for (_x, _y, rgba) in image.pixels() {
            let rgba = rgba.0.map(|i| i as f32 / 255.0);
            let rgb = Vec3A::from_array([rgba[0], rgba[1], rgba[2]]);
            let lab = rgb_to_lab(rgb);

            average_color += lab / (image.width() as f32 * image.height() as f32);
        }

        let image = image.resize_exact(
            TILE_SIZE,
            TILE_SIZE,
            image::imageops::FilterType::CatmullRom,
        );

        Self {
            image,
            average_color,
        }
    }
}

impl KdPoint for PaletteEntry {
    type Scalar = f32;
    type Dim = typenum::U3;

    fn at(&self, i: usize) -> Self::Scalar {
        self.average_color[i]
    }
}

fn get_tiles(image: &DynamicImage) -> (Vec<PaletteEntry>, u32, u32) {
    let num_tiles_x = image.width() / TILE_SIZE;
    let num_tiles_y = image.height() / TILE_SIZE;

    let mut tiles = Vec::with_capacity((num_tiles_x * num_tiles_y) as usize);

    for tile_y in 0..num_tiles_y {
        for tile_x in 0..num_tiles_x {
            let tile = image.crop_imm(tile_x * TILE_SIZE, tile_y * TILE_SIZE, TILE_SIZE, TILE_SIZE);

            let palette_entry = PaletteEntry::new(tile);
            tiles.push(palette_entry);
        }
    }

    (tiles, num_tiles_x, num_tiles_y)
}

fn main() {
    let start = std::time::Instant::now();

    println!(
        "({:.4} s) reading palette images...",
        start.elapsed().as_secs_f64()
    );

    let palette_paths: std::io::Result<Vec<_>> =
        std::fs::read_dir(std::env::current_dir().unwrap().join("palette"))
            .unwrap()
            .map(|e| e.map(|e| e.path()))
            .collect();
    let palette_paths = palette_paths.unwrap();

    let palette: image::ImageResult<Vec<_>> =
        palette_paths.into_par_iter().map(image::open).collect();
    let palette = palette.unwrap();

    println!(
        "({:.4} s) creating palette entries from palette images...",
        start.elapsed().as_secs_f64()
    );

    let palette: Vec<_> = palette.into_par_iter().map(PaletteEntry::new).collect();

    println!(
        "({:.4} s) building kd tree...",
        start.elapsed().as_secs_f64()
    );

    let kdtree = KdTree::par_build_by_ordered_float(palette);

    println!(
        "({:.4} s) reading mosaic image...",
        start.elapsed().as_secs_f64()
    );

    let mut mosaic = image::open(std::env::current_dir().unwrap().join("mosaic.jpg")).unwrap();
    mosaic = mosaic.crop_imm(
        0,
        0,
        (mosaic.width() / TILE_SIZE) * TILE_SIZE,
        (mosaic.height() / TILE_SIZE) * TILE_SIZE,
    );

    println!(
        "({:.4} s) creating tiles & palette entries from mosaic image...",
        start.elapsed().as_secs_f64()
    );

    let (tiles, num_tiles_x, num_tiles_y) = get_tiles(&mosaic);
    assert_eq!(tiles.len() as u32, num_tiles_x * num_tiles_y);

    println!(
        "({:.4} s) mapping tiles to palette images...",
        start.elapsed().as_secs_f64()
    );

    let mut mapped_tiles: Vec<&PaletteEntry> = Vec::with_capacity(tiles.len());

    for tile in &tiles {
        let closest_match = kdtree.nearest(tile).unwrap();
        mapped_tiles.push(closest_match.item);
    }

    println!(
        "({:.4} s) building final image...",
        start.elapsed().as_secs_f64()
    );

    let mut final_image = RgbaImage::new(mosaic.width(), mosaic.height());

    for tile_y in 0..num_tiles_y {
        for tile_x in 0..num_tiles_x {
            let base_pixel_x = tile_x * TILE_SIZE;
            let base_pixel_y = tile_y * TILE_SIZE;

            let tile_image = &mapped_tiles[(tile_x + tile_y * num_tiles_x) as usize].image;

            for offset_y in 0..tile_image.height() {
                for offset_x in 0..tile_image.width() {
                    final_image[(base_pixel_x + offset_x, base_pixel_y + offset_y)] =
                        tile_image.get_pixel(offset_x, offset_y);
                }
            }
        }
    }

    println!(
        "({:.4} s) writing to result.png...",
        start.elapsed().as_secs_f64()
    );

    final_image
        .save(std::env::current_dir().unwrap().join("result.png"))
        .unwrap();

    println!("({:.4} s) done!", start.elapsed().as_secs_f64());
}

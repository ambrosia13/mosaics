use glam::{Mat3A, Mat4, Vec3A};
use image::RgbImage;
use kd_tree::{KdPoint, KdTree};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

const TILE_SIZE: u32 = 32;

const XYZ_MATRIX: Mat3A = Mat3A::from_cols_array_2d(&[
    [0.4124564, 0.2126729, 0.0193339],
    [0.3575761, 0.7151522, 0.119_192],
    [0.1804375, 0.0721750, 0.9503041],
]);

fn srgb_to_linear(rgb: Vec3A) -> Vec3A {
    Vec3A::select(
        rgb.cmple(Vec3A::splat(0.04045)),
        rgb / 12.92,
        ((rgb + 0.055) / 1.055).powf(2.4),
    )
}

fn rgb_to_xyz(rgb: Vec3A) -> Vec3A {
    XYZ_MATRIX * rgb
}

fn xyz_to_lab(xyz: Vec3A) -> Vec3A {
    fn f(xyz: Vec3A) -> Vec3A {
        Vec3A::select(
            xyz.cmpgt(Vec3A::splat(0.008856)),
            Vec3A::new(xyz.x.cbrt(), xyz.y.cbrt(), xyz.z.cbrt()),
            7.787 * xyz + 16.0 / 116.0,
        )
    }

    let n = Vec3A::new(0.95047, 1.0, 1.08883);
    let f = f(xyz / n);

    Vec3A::new(116.0 * f.y - 16.0, 500.0 * (f.x - f.y), 200.0 * (f.y - f.z))
}

fn rgb_to_lab(rgb: Vec3A) -> Vec3A {
    xyz_to_lab(rgb_to_xyz(srgb_to_linear(rgb)))
}

struct PaletteEntry {
    image: RgbImage,
    average_color: Vec3A,
}

impl PaletteEntry {
    pub fn average_color(image: &RgbImage) -> Vec3A {
        let contribution = image.width() as f32 * image.height() as f32;

        image
            .pixels()
            .map(|p| {
                let rgb = Vec3A::from_array(p.0.map(|i| i as f32 / 255.0));
                rgb_to_lab(rgb)
            })
            .sum::<Vec3A>()
            / contribution
    }

    pub fn par_average_color(image: &RgbImage) -> Vec3A {
        let contribution = image.width() as f32 * image.height() as f32;

        image
            .par_pixels()
            .map(|p| {
                let rgb = Vec3A::from_array(p.0.map(|i| i as f32 / 255.0));
                rgb_to_lab(rgb)
            })
            .reduce(|| Vec3A::ZERO, |a, b| a + b)
            / contribution
    }

    pub fn new(image: RgbImage) -> Self {
        let average_color = Self::par_average_color(&image);

        let image = image::imageops::resize(
            &image,
            TILE_SIZE,
            TILE_SIZE,
            image::imageops::FilterType::CatmullRom,
        );

        Self {
            image,
            average_color,
        }
    }

    pub fn passthrough(image: RgbImage, average_color: Vec3A) -> Self {
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

fn get_tiles(image: &RgbImage) -> (Vec<PaletteEntry>, u32, u32) {
    // https://en.wikipedia.org/wiki/Ordered_dithering#Threshold_map
    // divide by max value, subtract half of max value to get normalized
    // bayer matrix can be transposed or rotated to the same effect
    // let bayer4: Mat4 = Mat4::from_cols_array_2d(&[
    //     [0.0, 8.0, 2.0, 10.0],
    //     [12.0, 4.0, 14.0, 6.0],
    //     [3.0, 11.0, 1.0, 9.0],
    //     [15.0, 7.0, 13.0, 5.0],
    // ]) / 16.0;

    let num_tiles_x = image.width() / TILE_SIZE;
    let num_tiles_y = image.height() / TILE_SIZE;

    let tiles: Vec<_> = (0..(num_tiles_x * num_tiles_y))
        .into_par_iter()
        .map(|i| {
            let tile_y = i / num_tiles_x;
            let tile_x = i % num_tiles_x;

            let tile = image::imageops::crop_imm(
                image,
                tile_x * TILE_SIZE,
                tile_y * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
            )
            .to_image();

            let bayer_x = tile_x % 4;
            let bayer_y = tile_y % 4;

            // let dither = bayer4.col(bayer_x as usize)[bayer_y as usize] - 0.5;

            let average_color = PaletteEntry::average_color(&tile);

            PaletteEntry::passthrough(tile, average_color)
        })
        .collect();

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

    let palette: image::ImageResult<Vec<_>> = palette_paths
        .into_par_iter()
        .map(image::open)
        .map(|i| i.map(|i| i.into_rgb8()))
        .collect();
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

    let mosaic = image::open(std::env::current_dir().unwrap().join("mosaic.jpg")).unwrap();
    let mosaic = mosaic
        .crop_imm(
            0,
            0,
            (mosaic.width() / TILE_SIZE) * TILE_SIZE,
            (mosaic.height() / TILE_SIZE) * TILE_SIZE,
        )
        .into_rgb8();

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

    let mut final_image = RgbImage::new(mosaic.width(), mosaic.height());

    for tile_y in 0..num_tiles_y {
        for tile_x in 0..num_tiles_x {
            let base_pixel_x = tile_x * TILE_SIZE;
            let base_pixel_y = tile_y * TILE_SIZE;

            let tile_image = &mapped_tiles[(tile_x + tile_y * num_tiles_x) as usize].image;

            for offset_y in 0..tile_image.height() {
                for offset_x in 0..tile_image.width() {
                    final_image[(base_pixel_x + offset_x, base_pixel_y + offset_y)] =
                        tile_image[(offset_x, offset_y)];
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

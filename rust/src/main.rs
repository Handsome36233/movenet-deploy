#![allow(clippy::manual_retain)]

use std::path::PathBuf;
use std::env;
use image::{imageops::FilterType, GenericImageView, Rgba, ImageBuffer};
use ndarray::Array;
use ort::{inputs, CUDAExecutionProvider, Session, SessionOutputs};
use raqote::{DrawOptions, DrawTarget, PathBuilder, SolidSource, Source};
use std::path::Path;

fn draw_keypoints(image: &mut DrawTarget, x: f32, y: f32, score: f32, conf_threshold: f32) {
    if score > conf_threshold {
        // 创建一个圆圈
        let mut pb = PathBuilder::new();
        pb.arc(x as f32, y as f32, 2.0, 0.0, 2.0 * std::f32::consts::PI);
        let path = pb.finish();

        // 设置颜色
        let color = SolidSource {
            r: 0,
            g: 255,
            b: 0,
            a: 255,
        };
        // 绘制圆圈
        image.stroke(&path, &Source::Solid(color), &raqote::StrokeStyle::default(), &raqote::DrawOptions::new());
    }
}

fn save_draw_target_as_png(draw_target: &DrawTarget, file_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let data = draw_target.get_data(); // 获取像素数据
    let width = draw_target.width() as u32;
    let height = draw_target.height() as u32;

    // 创建ImageBuffer来保存RGBA图像数据
    let mut img: ImageBuffer<Rgba<u8>, Vec<u8>> = ImageBuffer::new(width, height);

    // 将 DrawTarget 的 ARGB 像素数据转换为 RGBA 格式
    for (x, y, pixel) in img.enumerate_pixels_mut() {
        let argb = data[(y * width + x) as usize];
        let a = ((argb >> 24) & 0xff) as u8;
        let r = ((argb >> 16) & 0xff) as u8;
        let g = ((argb >> 8) & 0xff) as u8;
        let b = (argb & 0xff) as u8;

        *pixel = Rgba([r, g, b, a]); // 转换为 RGBA
    }

    // 保存图像
    img.save(Path::new(file_path))?;
    Ok(())
}


#[show_image::main]
fn main() -> ort::Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 3 {
        eprintln!("Usage: {} <model_path> <image_path>", args[0]);
        std::process::exit(1);
    }

    let model_path = PathBuf::from(&args[1]);
    let image_path = PathBuf::from(&args[2]);
	tracing_subscriber::fmt::init();

	ort::init()
		.with_execution_providers([CUDAExecutionProvider::default().build()])
		.commit()?;

	let input_width: u32 = 256;
	let input_height: u32 = 256;
    let conf_threshold = 0.3;
	let original_img = image::open(image_path).unwrap();
	let (img_width, img_height) = (original_img.width(), original_img.height());
	let img = original_img.resize_exact(input_width, input_height, FilterType::CatmullRom);
	let mut input = Array::zeros((1, 3, input_height as usize, input_width as usize));
	for pixel in img.pixels() {
		let x = pixel.0 as _;
		let y = pixel.1 as _;
		let [r, g, b, _] = pixel.2.0;
		input[[0, 0, y, x]] = r;
		input[[0, 1, y, x]] = g;
		input[[0, 2, y, x]] = b;
	}

	let model = Session::builder()?.commit_from_file(model_path)?;
	let input_shape = 	model.inputs[0].input_type.tensor_dimensions();
	let input_name = &model.inputs[0].name;
	let output_shape = model.outputs[0].output_type.tensor_dimensions();
	let output_name = &model.outputs[0].name;
	println!("input_shape: {:?}, input_name: {}", input_shape, input_name);
	println!("output_shape: {:?}, output_name: {}", output_shape, output_name);
	let mut draw_target = DrawTarget::new(img_width as i32, img_height as i32);
	// 遍历图像每个像素并复制到 DrawTarget
    for y in 0..img_height {
        for x in 0..img_width {
            let pixel = original_img.get_pixel(x, y);
            let color = raqote::SolidSource {
                r: pixel[0],
                g: pixel[1],
                b: pixel[2],
                a: pixel[3],
            };
            draw_target.fill_rect(x as f32, y as f32, 1.0, 1.0, &Source::Solid(color), &DrawOptions::new());
        }
    }

	let outputs: SessionOutputs = model.run(inputs![input_name => input.view()]?)?;
	let output: ndarray::ArrayBase<ndarray::ViewRepr<&f32>, ndarray::Dim<ndarray::IxDynImpl>> = outputs[output_name as &str].try_extract_tensor::<f32>().unwrap();
	for i in 0..output.shape()[2] {
		// 每个 keypoint 的数据都包含 3 个值：y, x, score
		let y = output[[0, 0, i, 0]] * img_height as f32;
		let x = output[[0, 0, i, 1]] * img_width as f32;
		let score = output[[0, 0, i, 2]];
		
		if score > conf_threshold {
			println!("score: {}, x: {}, y: {}", score, x, y);
			draw_keypoints(&mut draw_target, x, y, score, conf_threshold);
		}
	}
	save_draw_target_as_png(&draw_target, "output.png").expect("Failed to save image");

	Ok(())
}

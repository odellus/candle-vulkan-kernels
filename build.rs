use std::env;
use std::fs;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=shaders/");

    let out_dir = env::var("OUT_DIR")?;
    let shaders_dir = Path::new("shaders");
    let dest_dir = Path::new(&out_dir).join("shaders");

    fs::create_dir_all(&dest_dir)?;

    if shaders_dir.exists() {
        for entry in fs::read_dir(shaders_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("comp") {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap();
                let output_path = dest_dir.join(format!("{}.spv", stem));

                println!("cargo:warning=Compiling shader: {}", path.display());

                let compiler =
                    shaderc::Compiler::new().ok_or("Failed to create shader compiler")?;

                let mut options =
                    shaderc::CompileOptions::new().ok_or("Failed to create compile options")?;

                options.set_optimization_level(shaderc::OptimizationLevel::Performance);

                let source = fs::read_to_string(&path)?;
                let result = compiler.compile_into_spirv(
                    &source,
                    shaderc::ShaderKind::Compute,
                    &path.to_string_lossy(),
                    "main",
                    Some(&options),
                )?;

                fs::write(&output_path, result.as_binary_u8())?;
                println!(
                    "cargo:warning=Compiled {} -> {}",
                    stem,
                    output_path.display()
                );
            }
        }
    } else {
        println!("cargo:warning=shaders directory not found!");
    }

    println!("cargo:rustc-env=SHADER_DIR={}", out_dir);
    Ok(())
}

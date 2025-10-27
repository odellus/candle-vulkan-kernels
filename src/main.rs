use anyhow::Result;
use ash::{vk, Entry};
use log::info;
use std::ffi::CString;

fn main() -> Result<()> {
    env_logger::init();
    info!("Starting Candle Vulkan Demo - Simple Version");

    // Initialize Vulkan
    info!("Creating Vulkan context...");
    let entry = unsafe { Entry::load()? };

    // Create instance
    let app_name = CString::new("candle-vulkan-demo")?;
    let engine_name = CString::new("Candle")?;

    let app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(0)
        .engine_name(&engine_name)
        .engine_version(0)
        .api_version(vk::API_VERSION_1_2);

    let layer_names: Vec<*const i8> = vec![];
    let extension_names: Vec<*const i8> = vec![];

    let create_info = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_layer_names(&layer_names)
        .enabled_extension_names(&extension_names);

    let instance = unsafe { entry.create_instance(&create_info, None)? };
    info!("✅ Vulkan instance created");

    // Select physical device
    let physical_devices = unsafe { instance.enumerate_physical_devices()? };
    let physical_device = physical_devices[0];

    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let device_name =
        unsafe { std::ffi::CStr::from_ptr(properties.device_name.as_ptr()).to_string_lossy() };
    info!("Device: {}", device_name);
    info!(
        "API Version: {}.{}.{}",
        vk::api_version_major(properties.api_version),
        vk::api_version_minor(properties.api_version),
        vk::api_version_patch(properties.api_version)
    );

    // Find compute queue family
    let queue_family_properties =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let compute_queue_family = queue_family_properties
        .iter()
        .position(|qp| qp.queue_flags.contains(vk::QueueFlags::COMPUTE))
        .ok_or_else(|| anyhow::anyhow!("No compute queue family found"))?
        as u32;

    // Create device
    let queue_priorities = [1.0f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(compute_queue_family)
        .queue_priorities(&queue_priorities);

    let device_create_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(std::slice::from_ref(&queue_create_info));

    let device = unsafe { instance.create_device(physical_device, &device_create_info, None)? };
    info!("✅ Vulkan device created");

    // Get compute queue
    let compute_queue = unsafe { device.get_device_queue(compute_queue_family, 0) };
    info!("✅ Compute queue obtained");

    // Create command pool
    let command_pool_create_info = vk::CommandPoolCreateInfo::default()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(compute_queue_family);

    let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None)? };
    info!("✅ Command pool created");

    // Load compiled shader
    let shader_dir = env!("SHADER_DIR");
    let shader_path = std::path::Path::new(shader_dir).join("shaders/add.spv");
    let shader_code = std::fs::read(&shader_path)?;
    info!("Loaded shader from: {:?}", shader_path);

    // Ensure alignment
    let shader_u32: Vec<u32> = shader_code
        .chunks(4)
        .map(|chunk| {
            u32::from_le_bytes([
                chunk[0],
                chunk.get(1).copied().unwrap_or(0),
                chunk.get(2).copied().unwrap_or(0),
                chunk.get(3).copied().unwrap_or(0),
            ])
        })
        .collect();

    let create_info = vk::ShaderModuleCreateInfo::default().code(&shader_u32);

    let shader_module = unsafe { device.create_shader_module(&create_info, None)? };
    info!("✅ Shader module created");

    // Create descriptor set layout
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    let descriptor_set_layout_info =
        vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);

    let descriptor_set_layout =
        unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_info, None)? };
    info!("✅ Descriptor set layout created");

    // Create pipeline layout with push constants
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(5 * std::mem::size_of::<u32>() as u32);

    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(std::slice::from_ref(&descriptor_set_layout))
        .push_constant_ranges(std::slice::from_ref(&push_constant_range));

    let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_info, None)? };

    // Create compute pipeline
    let entry_point = CString::new("main")?;
    let stage_info = vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(&entry_point);

    let pipeline_info = vk::ComputePipelineCreateInfo::default()
        .stage(stage_info)
        .layout(pipeline_layout);

    let pipelines = unsafe {
        device
            .create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
            .map_err(|(_, e)| e)?
    };
    let pipeline = pipelines[0];
    info!("✅ Compute pipeline created");

    // Create buffers for matrices (example: 256x256 matrices)
    let rows = 256u32;
    let cols = 256u32;
    let buffer_size = (rows * cols) as u64 * std::mem::size_of::<f32>() as u64;

    let create_buffer = |usage: vk::BufferUsageFlags| -> Result<(vk::Buffer, vk::DeviceMemory)> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };

        let memory_type_index = find_memory_type(
            mem_requirements.memory_type_bits,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            &memory_properties,
        )?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(memory_type_index);

        let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
        unsafe { device.bind_buffer_memory(buffer, memory, 0)? };

        Ok((buffer, memory))
    };

    let (buffer_a, memory_a) = create_buffer(vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let (buffer_b, memory_b) = create_buffer(vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let (buffer_c, memory_c) = create_buffer(vk::BufferUsageFlags::STORAGE_BUFFER)?;
    info!("✅ Buffers created");

    // Initialize input matrices
    unsafe {
        let ptr =
            device.map_memory(memory_a, 0, buffer_size, vk::MemoryMapFlags::empty())? as *mut f32;
        for i in 0..(rows * cols) as usize {
            *ptr.add(i) = i as f32;
        }
        device.unmap_memory(memory_a);

        let ptr =
            device.map_memory(memory_b, 0, buffer_size, vk::MemoryMapFlags::empty())? as *mut f32;
        for i in 0..(rows * cols) as usize {
            *ptr.add(i) = (i * 2) as f32;
        }
        device.unmap_memory(memory_b);
    }
    info!("✅ Input matrices initialized");

    // Create descriptor pool
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::STORAGE_BUFFER)
        .descriptor_count(3);

    let pool_info = vk::DescriptorPoolCreateInfo::default()
        .pool_sizes(std::slice::from_ref(&pool_size))
        .max_sets(1);

    let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };

    // Allocate descriptor set
    let alloc_info = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(descriptor_pool)
        .set_layouts(std::slice::from_ref(&descriptor_set_layout));

    let descriptor_sets = unsafe { device.allocate_descriptor_sets(&alloc_info)? };
    let descriptor_set = descriptor_sets[0];

    // Update descriptor sets
    let buffer_info_a = vk::DescriptorBufferInfo::default()
        .buffer(buffer_a)
        .offset(0)
        .range(buffer_size);

    let buffer_info_b = vk::DescriptorBufferInfo::default()
        .buffer(buffer_b)
        .offset(0)
        .range(buffer_size);

    let buffer_info_c = vk::DescriptorBufferInfo::default()
        .buffer(buffer_c)
        .offset(0)
        .range(buffer_size);

    let write_sets = [
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info_a)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info_b)),
        vk::WriteDescriptorSet::default()
            .dst_set(descriptor_set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info_c)),
    ];

    unsafe { device.update_descriptor_sets(&write_sets, &[]) };
    info!("✅ Descriptor sets configured");

    // Create and record command buffer
    let command_buffer_info = vk::CommandBufferAllocateInfo::default()
        .command_pool(command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);

    let command_buffers = unsafe { device.allocate_command_buffers(&command_buffer_info)? };
    let command_buffer = command_buffers[0];

    let begin_info =
        vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe {
        device.begin_command_buffer(command_buffer, &begin_info)?;

        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
        device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_layout,
            0,
            std::slice::from_ref(&descriptor_set),
            &[],
        );

        // Push constants: rows, cols, stride_a, stride_b, stride_c
        let push_constants = [rows, cols, cols, cols, cols];
        device.cmd_push_constants(
            command_buffer,
            pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::cast_slice(&push_constants),
        );

        // Dispatch: (rows * cols) / 256 workgroups
        let workgroups = ((rows * cols) + 255) / 256;
        device.cmd_dispatch(command_buffer, workgroups, 1, 1);

        device.end_command_buffer(command_buffer)?;
    }

    // Submit and wait
    info!("Submitting compute command...");
    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

    unsafe {
        device.queue_submit(
            compute_queue,
            std::slice::from_ref(&submit_info),
            vk::Fence::null(),
        )?;
        device.queue_wait_idle(compute_queue)?;
    }

    info!("✅ Compute operation completed!");

    // Verify results
    unsafe {
        let ptr =
            device.map_memory(memory_c, 0, buffer_size, vk::MemoryMapFlags::empty())? as *const f32;
        let mut correct = true;
        for i in 0..10 {
            let expected = (i as f32) + (i * 2) as f32;
            let actual = *ptr.add(i);
            info!("Result[{}]: {} (expected: {})", i, actual, expected);
            if (actual - expected).abs() > 0.001 {
                correct = false;
            }
        }
        device.unmap_memory(memory_c);

        if correct {
            info!("✅ Results verified correctly!");
        } else {
            info!("❌ Results verification failed!");
        }
    }

    // Cleanup
    unsafe {
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_descriptor_set_layout(descriptor_set_layout, None);
        device.destroy_shader_module(shader_module, None);
        device.free_command_buffers(command_pool, &[command_buffer]);
        device.destroy_command_pool(command_pool, None);
        device.destroy_buffer(buffer_a, None);
        device.destroy_buffer(buffer_b, None);
        device.destroy_buffer(buffer_c, None);
        device.free_memory(memory_a, None);
        device.free_memory(memory_b, None);
        device.free_memory(memory_c, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    info!("Demo completed successfully!");
    Ok(())
}

fn find_memory_type(
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
    mem_properties: &vk::PhysicalDeviceMemoryProperties,
) -> Result<u32> {
    for i in 0..mem_properties.memory_type_count {
        if (type_filter & (1 << i)) != 0
            && mem_properties.memory_types[i as usize]
                .property_flags
                .contains(properties)
        {
            return Ok(i);
        }
    }
    Err(anyhow::anyhow!("Failed to find suitable memory type"))
}

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::{DeviceEvent, ElementState, Event, KeyEvent, WindowEvent},
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Fullscreen, Window, WindowBuilder},
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    time: f32,
    fractal_type: i32,
    camera_pos: [f32; 3],
    power: f32,
    camera_forward: [f32; 3],
    color_shift: f32,
    camera_right: [f32; 3],
    zoom_depth: f32,
    camera_up: [f32; 3],
    _pad: f32,
}

struct Camera {
    position: Vec3,
    yaw: f32,
    pitch: f32,
    speed: f32,
    sensitivity: f32,
}

impl Camera {
    fn new() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 3.0),
            yaw: -90.0_f32.to_radians(),
            pitch: 0.0,
            speed: 2.0,
            sensitivity: 0.003,
        }
    }

    fn forward(&self) -> Vec3 {
        Vec3::new(
            self.yaw.cos() * self.pitch.cos(),
            self.pitch.sin(),
            self.yaw.sin() * self.pitch.cos(),
        )
        .normalize()
    }

    fn right(&self) -> Vec3 {
        self.forward().cross(Vec3::Y).normalize()
    }

    fn up(&self) -> Vec3 {
        self.right().cross(self.forward()).normalize()
    }

    fn process_mouse(&mut self, dx: f64, dy: f64) {
        self.yaw += dx as f32 * self.sensitivity;
        self.pitch -= dy as f32 * self.sensitivity;
        self.pitch = self.pitch.clamp(-89.0_f32.to_radians(), 89.0_f32.to_radians());
    }
}

struct InputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    boost: bool,
}

impl InputState {
    fn new() -> Self {
        Self {
            forward: false,
            backward: false,
            left: false,
            right: false,
            up: false,
            down: false,
            boost: false,
        }
    }
}

struct State {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    camera: Camera,
    input: InputState,
    start_time: Instant,
    last_frame: Instant,
    fractal_type: i32,
    power: f32,
    color_shift: f32,
    zoom_depth: f32,
    cursor_grabbed: bool,
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let uniforms = Uniforms {
            resolution: [size.width as f32, size.height as f32],
            time: 0.0,
            fractal_type: 0,
            camera_pos: [0.0, 0.0, 3.0],
            power: 8.0,
            camera_forward: [0.0, 0.0, -1.0],
            color_shift: 0.0,
            camera_right: [1.0, 0.0, 0.0],
            zoom_depth: 0.0,
            camera_up: [0.0, 1.0, 0.0],
            _pad: 0.0,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[uniforms]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
            label: Some("uniform_bind_group"),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            uniform_buffer,
            uniform_bind_group,
            camera: Camera::new(),
            input: InputState::new(),
            start_time: Instant::now(),
            last_frame: Instant::now(),
            fractal_type: 0,
            power: 8.0,
            color_shift: 0.0,
            zoom_depth: 0.0,
            cursor_grabbed: false,
        }
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn update(&mut self) {
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Movement
        let speed = if self.input.boost {
            self.camera.speed * 5.0
        } else {
            self.camera.speed
        };

        let mut velocity = Vec3::ZERO;

        if self.input.forward {
            velocity += self.camera.forward();
        }
        if self.input.backward {
            velocity -= self.camera.forward();
        }
        if self.input.right {
            velocity += self.camera.right();
        }
        if self.input.left {
            velocity -= self.camera.right();
        }
        if self.input.up {
            velocity += Vec3::Y;
        }
        if self.input.down {
            velocity -= Vec3::Y;
        }

        if velocity.length_squared() > 0.0 {
            velocity = velocity.normalize() * speed * dt;
            self.camera.position += velocity;
        }

        // Update uniforms
        let uniforms = Uniforms {
            resolution: [self.size.width as f32, self.size.height as f32],
            time: self.start_time.elapsed().as_secs_f32(),
            fractal_type: self.fractal_type,
            camera_pos: self.camera.position.to_array(),
            power: self.power,
            camera_forward: self.camera.forward().to_array(),
            color_shift: self.color_shift,
            camera_right: self.camera.right().to_array(),
            zoom_depth: self.zoom_depth,
            camera_up: self.camera.up().to_array(),
            _pad: 0.0,
        };

        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = Arc::new(
        WindowBuilder::new()
            .with_title("Multiverse Explorer")
            .with_fullscreen(Some(Fullscreen::Borderless(None)))
            .build(&event_loop)
            .unwrap(),
    );

    let mut state = pollster::block_on(State::new(window.clone()));

    // Grab cursor for FPS controls
    let _ = window.set_cursor_grab(CursorGrabMode::Confined);
    window.set_cursor_visible(false);
    state.cursor_grabbed = true;

    event_loop
        .run(move |event, target| match event {
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } => {
                if state.cursor_grabbed {
                    state.camera.process_mouse(delta.0, delta.1);
                }
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => match event {
                WindowEvent::CloseRequested => target.exit(),
                WindowEvent::Resized(physical_size) => {
                    state.resize(*physical_size);
                }
                WindowEvent::KeyboardInput {
                    event:
                        KeyEvent {
                            physical_key: PhysicalKey::Code(keycode),
                            state: key_state,
                            ..
                        },
                    ..
                } => {
                    let pressed = *key_state == ElementState::Pressed;
                    match keycode {
                        KeyCode::KeyW => state.input.forward = pressed,
                        KeyCode::KeyS => state.input.backward = pressed,
                        KeyCode::KeyA => state.input.left = pressed,
                        KeyCode::KeyD => state.input.right = pressed,
                        KeyCode::Space => state.input.up = pressed,
                        KeyCode::ControlLeft | KeyCode::KeyC => state.input.down = pressed,
                        KeyCode::ShiftLeft => state.input.boost = pressed,
                        KeyCode::Escape => {
                            if pressed {
                                if state.cursor_grabbed {
                                    let _ = window.set_cursor_grab(CursorGrabMode::None);
                                    window.set_cursor_visible(true);
                                    state.cursor_grabbed = false;
                                } else {
                                    target.exit();
                                }
                            }
                        }
                        KeyCode::Digit1 => {
                            if pressed {
                                state.fractal_type = 0;
                                state.power = 8.0;
                            }
                        }
                        KeyCode::Digit2 => {
                            if pressed {
                                state.fractal_type = 1;
                                state.power = 2.0;
                            }
                        }
                        KeyCode::Digit3 => {
                            if pressed {
                                state.fractal_type = 2;
                                state.power = 8.0;
                            }
                        }
                        KeyCode::KeyQ => {
                            if pressed {
                                state.power = (state.power - 0.5).max(2.0);
                            }
                        }
                        KeyCode::KeyE => {
                            if pressed {
                                state.power = (state.power + 0.5).min(12.0);
                            }
                        }
                        KeyCode::KeyZ => {
                            if pressed {
                                state.color_shift = (state.color_shift - 0.1) % 1.0;
                            }
                        }
                        KeyCode::KeyX => {
                            if pressed {
                                state.color_shift = (state.color_shift + 0.1) % 1.0;
                            }
                        }
                        KeyCode::KeyR => {
                            if pressed {
                                state.camera.position = Vec3::new(0.0, 0.0, 3.0);
                                state.camera.yaw = -90.0_f32.to_radians();
                                state.camera.pitch = 0.0;
                                state.zoom_depth = 0.0;
                            }
                        }
                        _ => {}
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    // Scroll to zoom in/out
                    let scroll = match delta {
                        winit::event::MouseScrollDelta::LineDelta(_, y) => *y,
                        winit::event::MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.01,
                    };
                    state.zoom_depth = (state.zoom_depth + scroll * 0.2).max(0.0);
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    ..
                } => {
                    if !state.cursor_grabbed {
                        let _ = window.set_cursor_grab(CursorGrabMode::Confined);
                        window.set_cursor_visible(false);
                        state.cursor_grabbed = true;
                    }
                }
                WindowEvent::RedrawRequested => {
                    state.update();
                    match state.render() {
                        Ok(_) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => target.exit(),
                        Err(e) => eprintln!("{:?}", e),
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => {}
        })
        .unwrap();
}

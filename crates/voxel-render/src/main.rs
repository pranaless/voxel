use cgmath::{
    Deg, InnerSpace, Matrix, Matrix3, Matrix4, One, PerspectiveFov, SquareMatrix, Vector2, Vector3,
    Vector4, Zero,
};
use chunk::{
    render::{ChunkMesh, Face, Vertex},
    Sided,
};
use parking_lot::Mutex;
use std::time::{Duration, Instant};
use wgpu::{include_wgsl, BufferUsages, Features, TextureUsages};
use winit::{
    event::{DeviceEvent, Event, KeyboardInput, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub mod chunk;

pub struct RenderState {
    device: wgpu::Device,
    queue: wgpu::Queue,

    camera_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    chunk_mesh_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,

    surface: wgpu::Surface,
    pub window: Window,
    config: Mutex<wgpu::SurfaceConfiguration>,
    pub swapchain_format: wgpu::TextureFormat,
}
impl RenderState {
    pub fn resize(&self, width: u32, height: u32) {
        let mut config = self.config.lock();
        config.width = width;
        config.height = height;
        self.surface.configure(&self.device, &config);
    }
}

pub struct Camera {
    buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    pub transform: Matrix4<f64>,
}
impl Camera {
    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 128,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Camera {
            buffer,
            bind_group,
            transform: Matrix4::one(),
        }
    }

    pub fn update_viewport(&self, queue: &wgpu::Queue, viewport: Matrix4<f64>) {
        #[rustfmt::skip]
        const OPENGL_TO_WGPU_MATRIX: Matrix4<f64> = Matrix4::new(
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 0.5, 0.0,
            0.0, 0.0, 0.5, 1.0,
        );

        let viewport = OPENGL_TO_WGPU_MATRIX * viewport;
        let viewport: Matrix4<f32> = viewport.cast().unwrap();
        let viewport: &[f32; 16] = viewport.as_ref();
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(viewport));
    }

    #[inline]
    pub fn apply_transform(&mut self, delta: Matrix4<f64>) {
        self.transform = self.transform * delta;
    }

    pub fn commit_transform(&self, queue: &wgpu::Queue) {
        let transform = self.transform.invert().unwrap();
        let transform: Matrix4<f32> = transform.cast().unwrap();
        let transform: &[f32; 16] = transform.as_ref();
        queue.write_buffer(&self.buffer, 64, bytemuck::cast_slice(transform));
    }

    pub fn set_bind_group<'a>(&'a self, rpass: &mut wgpu::RenderPass<'a>, index: u32) {
        rpass.set_bind_group(index, &self.bind_group, &[]);
    }
}

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub struct PlayerInput {
    pub forward: bool,
    pub backward: bool,
    pub leftward: bool,
    pub rightward: bool,
    pub upward: bool,
    pub downward: bool,
}
impl PlayerInput {
    pub fn update(&mut self, scancode: u32, pressed: bool) {
        *(match scancode {
            17 => &mut self.forward,
            30 => &mut self.leftward,
            31 => &mut self.backward,
            32 => &mut self.rightward,
            42 => &mut self.downward,
            57 => &mut self.upward,
            _ => return,
        }) = pressed;
    }

    pub fn delta(&self) -> Option<Vector3<f64>> {
        let mut delta = Vector3::<i32>::zero();
        if self.leftward {
            delta.x -= 1;
        }
        if self.rightward {
            delta.x += 1;
        }
        if self.downward {
            delta.y -= 1;
        }
        if self.upward {
            delta.y += 1;
        }
        if self.backward {
            delta.z += 1;
        }
        if self.forward {
            delta.z -= 1;
        }
        if !delta.is_zero() {
            delta.cast()
        } else {
            None
        }
    }
}

pub fn translation(v: Vector3<f64>) -> Matrix4<f64> {
    let w = (1.0 + v.magnitude2()).sqrt();
    let c = (v / (w + 1.0)).extend(1.0);
    Matrix4::from_cols(
        c * v.x + Vector4::unit_x(),
        c * v.y + Vector4::unit_y(),
        c * v.z + Vector4::unit_z(),
        v.extend(w),
    )
}

fn main() {
    env_logger::init();

    let event_loop = EventLoop::new();
    let window = Window::new(&event_loop).unwrap();

    let state = pollster::block_on(async {
        let instance = wgpu::Instance::new(Default::default());
        let surface = unsafe { instance.create_surface(&window).unwrap() };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .unwrap();

        let camera_layout = Camera::layout(&device);
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let chunk_mesh_layout = ChunkMesh::layout(&device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_layout, &texture_layout, &chunk_mesh_layout],
            push_constant_ranges: &[],
        });

        let caps = surface.get_capabilities(&adapter);
        let config = Mutex::new(wgpu::SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: 0,
            height: 0,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: Vec::new(),
        });

        RenderState {
            device,
            queue,
            camera_layout,
            texture_layout,
            chunk_mesh_layout,
            pipeline_layout,
            surface,
            window,
            config,
            swapchain_format: caps.formats[0],
        }
    });

    let shader = state
        .device
        .create_shader_module(include_wgsl!("shader.wgsl"));

    let pipeline = state
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&state.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::LAYOUT],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(state.swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState {
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

    let mut depth = state.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Depth32Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let mut depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

    let mut camera = Camera::new(&state.device, &state.camera_layout);

    let mut viewport = PerspectiveFov {
        fovy: Deg(45.0).into(),
        aspect: 1.0,
        near: 0.1,
        far: 100.0,
    };
    camera.commit_transform(&state.queue);

    let mut input = PlayerInput::default();
    let mut tracking = false;

    const SIDE: f32 = 0.485_868_28;
    const STEP: f64 = 1.272_019_649_514_069;

    let mesh = {
        let mut chunk = [[[false; 16]; 16]; 16];
        for x in 0..16 {
            for z in 0..16 {
                let ym = ((x as f64 / 4.0).sin() * (z as f64 / 4.0).cos() * 6.0 + 8.0) as usize;
                for y in 0..ym {
                    chunk[x][y][z] = true;
                }
            }
        }
        let mut mesh = [[[Default::default(); 16]; 16]; 16];
        for x in 0..16 {
            for y in 0..16 {
                for z in 0..16 {
                    let s = chunk[x][y][z];
                    mesh[x][y][z] = Sided {
                        neg_x: Face(s && (x == 0 || !chunk[x - 1][y][z])),
                        pos_x: Face(s && (x == 15 || !chunk[x + 1][y][z])),
                        neg_y: Face(s && (y == 0 || !chunk[x][y - 1][z])),
                        pos_y: Face(s && (y == 15 || !chunk[x][y + 1][z])),
                        neg_z: Face(s && (z == 0 || !chunk[x][y][z - 1])),
                        pos_z: Face(s && (z == 15 || !chunk[x][y][z + 1])),
                    };
                }
            }
        }
        mesh
    };

    let mut chunks = Vec::new();

    let mut insert = |tr: Matrix4<f64>| -> Matrix4<f64> {
        let mut chunk = ChunkMesh::new(&state.device, &state.chunk_mesh_layout);
        chunk.update_transform(&state.queue, tr);
        chunk.update_mesh(&state.device, SIDE, &mesh);
        chunks.push(chunk);
        tr
    };

    let trs = Sided {
        neg_x: translation(Vector3::new(-STEP, 0.0, 0.0)),
        pos_x: translation(Vector3::new(STEP, 0.0, 0.0)),
        neg_y: translation(Vector3::new(0.0, -STEP, 0.0)),
        pos_y: translation(Vector3::new(0.0, STEP, 0.0)),
        neg_z: translation(Vector3::new(0.0, 0.0, -STEP)),
        pos_z: translation(Vector3::new(0.0, 0.0, STEP)),
    };

    let c0 = insert(Matrix4::one());
    let c1 = insert(c0 * trs.pos_x);
    let c2 = insert(c0 * trs.pos_z);
    let c3 = insert(c0 * trs.neg_x);
    let c4 = insert(c0 * trs.neg_z);
    let _c5 = insert(c1 * trs.pos_z);
    let _c6 = insert(c1 * trs.neg_z);
    let _c7 = insert(c2 * trs.pos_x);
    let _c8 = insert(c2 * trs.neg_x);
    let _c9 = insert(c3 * trs.neg_z);
    let _c10 = insert(c3 * trs.pos_z);
    let _c11 = insert(c4 * trs.neg_x);
    let _c12 = insert(c4 * trs.pos_x);

    let stone = include_bytes!("stone.png");
    let stone = image::load_from_memory(stone).unwrap().into_rgba8();

    let extent = stone.dimensions();
    let extent = wgpu::Extent3d {
        width: extent.0,
        height: extent.1,
        depth_or_array_layers: 1,
    };
    let texture = state.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: extent,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
        view_formats: &[],
    });
    state.queue.write_texture(
        wgpu::ImageCopyTexture {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &stone,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(4 * extent.width),
            rows_per_image: Some(extent.height),
        },
        wgpu::Extent3d {
            depth_or_array_layers: 1,
            ..extent
        },
    );

    let texture_view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: None,
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let texture_sampler = state.device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        ..Default::default()
    });

    let texture_bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &state.texture_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&texture_sampler),
            },
        ],
    });

    let mut time = Instant::now();
    let step = Duration::from_nanos(16_666_666);

    event_loop.run(move |event, _, ctrl| {
        let _ = (&depth,);

        match event {
            Event::NewEvents(StartCause::Init) => {
                time = Instant::now();
                *ctrl = ControlFlow::WaitUntil(time + step);
                state.window.request_redraw();
            }
            Event::NewEvents(StartCause::ResumeTimeReached { .. }) => {
                let delta_time = {
                    let prev = std::mem::replace(&mut time, Instant::now());
                    *ctrl = ControlFlow::WaitUntil(time + step);
                    (time - prev).as_secs_f64()
                };

                if let Some(delta) = input.delta() {
                    let delta = delta.normalize() * delta_time * STEP / 16.0 * 5.0;
                    camera.apply_transform(translation(delta));
                }
                camera.commit_transform(&state.queue);

                state.window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                state.resize(size.width, size.height);
                depth = state.device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: size.width,
                        height: size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Depth32Float,
                    usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                });
                depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());

                viewport.aspect = size.width as f64 / size.height as f64;
                camera.update_viewport(&state.queue, viewport.into());
            }
            Event::WindowEvent {
                event:
                    WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state, scancode, ..
                            },
                        ..
                    },
                ..
            } => input.update(
                scancode,
                matches!(state, winit::event::ElementState::Pressed),
            ),
            Event::WindowEvent {
                event: WindowEvent::CursorEntered { .. },
                ..
            } => tracking = true,
            Event::WindowEvent {
                event: WindowEvent::CursorLeft { .. },
                ..
            } => tracking = false,
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta },
                ..
            } if tracking => {
                let delta = 0.01 * Vector2::new(delta.0, -delta.1);
                camera.apply_transform({
                    let r = delta.magnitude2();
                    let delta = (delta - delta * r / 6.0).extend(1.0 - r).normalize();

                    let s = Vector3::unit_y().cross(delta).normalize();
                    let u = delta.cross(s);
                    Matrix4::from(Matrix3::from_cols(s, u, delta).transpose())
                });
            }
            Event::RedrawRequested(_) => {
                let frame = state.surface.get_current_texture().unwrap();
                let view = frame.texture.create_view(&Default::default());
                let mut encoder = state
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                {
                    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                store: true,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &depth_view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: true,
                            }),
                            stencil_ops: None,
                        }),
                    });
                    rpass.set_pipeline(&pipeline);
                    camera.set_bind_group(&mut rpass, 0);
                    rpass.set_bind_group(1, &texture_bind_group, &[]);
                    for ch in chunks.iter() {
                        ch.draw(&mut rpass, 2);
                    }
                }
                state.queue.submit(Some(encoder.finish()));
                frame.present();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => *ctrl = ControlFlow::Exit,
            _ => {}
        }
    });
}

use cgmath::{Deg, InnerSpace, Matrix4, One, PerspectiveFov, Vector2, Vector3, Zero};
use gltf::buffer::Source;
use gltf::Gltf;
use parking_lot::Mutex;
use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use voxel_render::mesh::InstanceTransformData;
use voxel_render::{camera::Camera, mesh::VertexData};
use wgpu::{include_wgsl, util::DeviceExt, Features, TextureUsages};
use wgpu::{IndexFormat, RenderPass};
use winit::{
    event::{DeviceEvent, Event, KeyboardInput, StartCause, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

pub struct RenderState {
    device: wgpu::Device,
    queue: wgpu::Queue,

    camera_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
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

pub struct CameraBindGroup(wgpu::BindGroup);
impl CameraBindGroup {
    pub fn layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[Camera::layout_entry(0)],
        })
    }

    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, camera: &Camera) -> Self {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[camera.entry(0)],
        });
        CameraBindGroup(bind_group)
    }

    pub fn set_bind_group<'a>(&'a self, rpass: &mut RenderPass<'a>, index: u32) {
        rpass.set_bind_group(index, &self.0, &[]);
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

struct BufferRef {
    buffer: Arc<wgpu::Buffer>,
    offset: u64,
}
impl BufferRef {
    pub fn slice(&self) -> wgpu::BufferSlice<'_> {
        self.buffer.slice(self.offset..)
    }
}

struct Mesh {
    index: Option<(IndexFormat, BufferRef)>,
    count: u32,
    instances: u32,
}
impl Mesh {
    pub fn draw<'a>(&'a self, rpass: &mut RenderPass<'a>) {
        if let Some((format, buffer)) = self.index.as_ref() {
            rpass.set_index_buffer(buffer.slice(), *format);
            rpass.draw_indexed(0..self.count, 0, 0..self.instances);
        } else {
            rpass.draw(0..self.count, 0..self.instances);
        }
    }
}

pub struct MeshRenderer {
    slots: Vec<u32>,
    buffers: Vec<BufferRef>,
    meshes: Vec<Mesh>,
}
impl MeshRenderer {
    pub fn draw<'a>(&'a self, rpass: &mut RenderPass<'a>) {
        self.buffers
            .chunks(self.slots.len())
            .zip(self.meshes.iter())
            .for_each(|(buffers, mesh)| {
                self.slots
                    .iter()
                    .zip(buffers)
                    .for_each(|(&slot, buffer)| rpass.set_vertex_buffer(slot, buffer.slice()));

                mesh.draw(rpass);
            });
    }
}

pub fn load_gltf<P: AsRef<Path>>(device: &wgpu::Device, path: P) {
    let gltf = Gltf::open(path).unwrap();
    let buffers = gltf
        .buffers()
        .map(|b| b.source())
        .map(|s| match s {
            Source::Bin => gltf.blob.as_deref().map(Cow::Borrowed),
            Source::Uri(uri) => {
                log::error!("Trying to load from \"{uri}\", which is not supported");
                None
            }
        })
        .collect::<Option<Vec<_>>>()
        .expect("failed to load all buffers");

    gltf.meshes().flat_map(|m| m.primitives()).for_each(|m| {
        // 1. gen vertex attrib layout
        // 2. upload required buffers
        // 3. assemble mesh
    });
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

        let camera_layout = CameraBindGroup::layout(&device);
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
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_layout, &texture_layout],
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
                buffers: &[VertexData::LAYOUT, InstanceTransformData::LAYOUT],
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

    let mut camera = Camera::new(
        &state.device,
        PerspectiveFov {
            fovy: Deg(45.0).into(),
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        },
        Matrix4::one(),
    );
    let camera_bind_group = CameraBindGroup::new(&state.device, &state.camera_layout, &camera);

    let mut input = PlayerInput::default();
    let mut tracking = false;

    const STEP: f64 = 1.272_019_649_514_069;

    // let mut mesh = load_gltf(&state.device, "res/cell.glb");
    // let mesh = mesh.pop().unwrap();

    let chunk_data = vec![InstanceTransformData::new(Matrix4::one())];

    let chunk_len = chunk_data.len();
    let chunk_data = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            usage: wgpu::BufferUsages::VERTEX,
            contents: bytemuck::cast_slice(&chunk_data),
        });

    println!("{}", chunk_len);

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
                    camera.apply_transform(Matrix4::from_translation(delta));
                    camera.commit(&state.queue);
                }

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

                camera.viewport.aspect = size.width as f64 / size.height as f64;
                camera.commit(&state.queue);
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
                camera.apply_rotation(0.01 * Vector2::new(delta.0, -delta.1));
                camera.commit(&state.queue);
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
                    camera_bind_group.set_bind_group(&mut rpass, 0);
                    rpass.set_bind_group(1, &texture_bind_group, &[]);
                    rpass.set_vertex_buffer(1, chunk_data.slice(..));
                    // for primitive in mesh.primitives.iter() {
                    //     primitive.draw(&mut rpass, chunk_len);
                    // }
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

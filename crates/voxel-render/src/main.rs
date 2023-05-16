use bytemuck::{Pod, Zeroable};
use parking_lot::Mutex;
use wgpu::{include_wgsl, util::DeviceExt, Features};
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::Window,
};

pub struct RenderState {
    pub window: Window,
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,

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

#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub uv: [f32; 2],
}
impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: 20,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 0,
                shader_location: 0,
            },
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 12,
                shader_location: 1,
            },
        ],
    };
}

const VERTICIES: &[Vertex] = &[
    Vertex {
        pos: [-1.0, -1.0, 0.0],
        uv: [0.0, 0.0],
    },
    Vertex {
        pos: [1.0, -1.0, 0.0],
        uv: [1.0, 0.0],
    },
    Vertex {
        pos: [0.0, 1.0, 0.0],
        uv: [0.5, 1.0],
    },
];

const INDICIES: &[u32] = &[0, 1, 2];

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

        let caps = surface.get_capabilities(&adapter);
        let config = Mutex::new(wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: caps.formats[0],
            width: 0,
            height: 0,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: Vec::new(),
        });
        RenderState {
            window,
            surface,
            device,
            queue,
            config,
            swapchain_format: caps.formats[0],
        }
    });

    let shader = state
        .device
        .create_shader_module(include_wgsl!("shader.wgsl"));

    let pipeline_layout = state
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });

    let pipeline = state
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
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
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

    let vertex = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            usage: wgpu::BufferUsages::VERTEX,
            contents: bytemuck::cast_slice(VERTICIES),
        });

    let index = state
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            usage: wgpu::BufferUsages::INDEX,
            contents: bytemuck::cast_slice(INDICIES),
        });

    event_loop.run(move |event, _, ctrl| match event {
        Event::WindowEvent {
            event: WindowEvent::Resized(size),
            ..
        } => {
            state.resize(size.width, size.height);
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
                    depth_stencil_attachment: None,
                });
                rpass.set_pipeline(&pipeline);
                rpass.set_vertex_buffer(0, vertex.slice(..));
                rpass.set_index_buffer(index.slice(..), wgpu::IndexFormat::Uint32);
                rpass.draw_indexed(0..3, 0, 0..1);
            }
            state.queue.submit(Some(encoder.finish()));
            frame.present();
        }
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => *ctrl = winit::event_loop::ControlFlow::Exit,
        _ => {}
    });
}

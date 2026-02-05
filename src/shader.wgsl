struct Uniforms {
    resolution: vec2<f32>,
    time: f32,
    fractal_type: i32,
    camera_pos: vec3<f32>,
    power: f32,
    camera_forward: vec3<f32>,
    color_shift: f32,
    camera_right: vec3<f32>,
    zoom_depth: f32,
    camera_up: vec3<f32>,
    _pad: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
    );

    var out: VertexOutput;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = positions[vertex_index];
    return out;
}

const MAX_STEPS: i32 = 256;
const MAX_DIST: f32 = 20.0;
const PI: f32 = 3.14159265359;

// ============================================================================
// NOISE FUNCTIONS - For infinite procedural detail
// ============================================================================

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var q = vec3<f32>(
        dot(p, vec3<f32>(127.1, 311.7, 74.7)),
        dot(p, vec3<f32>(269.5, 183.3, 246.1)),
        dot(p, vec3<f32>(113.5, 271.9, 124.6))
    );
    return fract(sin(q) * 43758.5453123);
}

fn hash31(p: vec3<f32>) -> f32 {
    let h = dot(p, vec3<f32>(127.1, 311.7, 74.7));
    return fract(sin(h) * 43758.5453123);
}

// Smooth 3D noise
fn noise3d(p: vec3<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);

    return mix(
        mix(
            mix(hash31(i + vec3<f32>(0.0, 0.0, 0.0)), hash31(i + vec3<f32>(1.0, 0.0, 0.0)), u.x),
            mix(hash31(i + vec3<f32>(0.0, 1.0, 0.0)), hash31(i + vec3<f32>(1.0, 1.0, 0.0)), u.x),
            u.y
        ),
        mix(
            mix(hash31(i + vec3<f32>(0.0, 0.0, 1.0)), hash31(i + vec3<f32>(1.0, 0.0, 1.0)), u.x),
            mix(hash31(i + vec3<f32>(0.0, 1.0, 1.0)), hash31(i + vec3<f32>(1.0, 1.0, 1.0)), u.x),
            u.y
        ),
        u.z
    );
}

// ============================================================================
// INFINITE DETAIL FRACTAL
// Uses FBM noise that adds new octaves as you zoom deeper
// ============================================================================

fn infinite_detail_fbm(p: vec3<f32>, zoom_level: f32) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var frequency: f32 = 1.0;
    var pos = p;

    // Number of octaves increases with zoom - MORE DETAIL as you zoom
    let base_octaves = 4;
    let zoom_octaves = i32(zoom_level * 2.0);
    let total_octaves = base_octaves + zoom_octaves;

    for (var i = 0; i < 20; i++) {  // Max 20 octaves
        if (i >= total_octaves) { break; }

        value += amplitude * (noise3d(pos * frequency) - 0.5);
        amplitude *= 0.5;
        frequency *= 2.0;

        // Rotate each octave slightly for more interesting patterns
        pos = vec3<f32>(
            pos.y * 1.1 + pos.z * 0.3,
            pos.z * 1.1 - pos.x * 0.3,
            pos.x * 1.1 + pos.y * 0.3
        );
    }

    return value;
}

// Base shape - sphere with fractal displacement
fn fractal_sphere_de(p: vec3<f32>, zoom_level: f32) -> vec2<f32> {
    // Base sphere
    let sphere_dist = length(p) - 1.0;

    // Add infinite detail displacement
    // The displacement amount stays visually constant as we zoom
    let detail = infinite_detail_fbm(p * exp(zoom_level), zoom_level);
    let displacement = detail * 0.3;

    let dist = sphere_dist + displacement;

    // Orbit trap for coloring - based on noise layers
    let trap = abs(detail) * 2.0 + length(p) * 0.2;

    return vec2<f32>(dist, trap);
}

// Mandelbulb with infinite detail overlay
fn mandelbulb_infinite_de(pos: vec3<f32>, power: f32, zoom_level: f32) -> vec2<f32> {
    var z = pos;
    var dr: f32 = 1.0;
    var r: f32 = 0.0;
    var orbit_trap: f32 = 1e10;

    for (var i = 0; i < 12; i++) {
        r = length(z);
        if (r > 2.0) { break; }

        let theta = acos(z.z / r);
        let phi = atan2(z.y, z.x);

        dr = pow(r, power - 1.0) * power * dr + 1.0;

        let zr = pow(r, power);
        let new_theta = theta * power;
        let new_phi = phi * power;

        z = zr * vec3<f32>(
            sin(new_theta) * cos(new_phi),
            sin(new_theta) * sin(new_phi),
            cos(new_theta)
        );
        z += pos;

        orbit_trap = min(orbit_trap, length(z));
    }

    // Base Mandelbulb distance
    var dist = 0.5 * log(r) * r / dr;

    // Add infinite detail on top of the Mandelbulb surface
    // This is the key - procedural detail that increases with zoom
    let detail_scale = exp(zoom_level);
    let detail = infinite_detail_fbm(pos * detail_scale, zoom_level);

    // Displacement decreases in world space but stays constant visually
    let displacement = detail * 0.1 / detail_scale;
    dist += displacement;

    // Mix orbit trap with noise for varied coloring
    let color_detail = infinite_detail_fbm(pos * detail_scale * 0.5, zoom_level * 0.5);
    orbit_trap = mix(orbit_trap, abs(color_detail) + 0.5, 0.3);

    return vec2<f32>(dist, orbit_trap);
}

// Main distance function
fn scene_de(p: vec3<f32>, zoom_level: f32) -> vec2<f32> {
    if (uniforms.fractal_type == 1) {
        return fractal_sphere_de(p, zoom_level);
    } else {
        return mandelbulb_infinite_de(p, uniforms.power, zoom_level);
    }
}

// ============================================================================
// RENDERING
// ============================================================================

fn cosmic_palette(t: f32, shift: f32) -> vec3<f32> {
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.0 + shift, 0.33 + shift, 0.67 + shift);
    return a + b * cos(6.28318 * (c * t + d));
}

fn calc_normal(p: vec3<f32>, zoom_level: f32) -> vec3<f32> {
    let e = 0.001;
    return normalize(vec3<f32>(
        scene_de(p + vec3<f32>(e, 0.0, 0.0), zoom_level).x - scene_de(p - vec3<f32>(e, 0.0, 0.0), zoom_level).x,
        scene_de(p + vec3<f32>(0.0, e, 0.0), zoom_level).x - scene_de(p - vec3<f32>(0.0, e, 0.0), zoom_level).x,
        scene_de(p + vec3<f32>(0.0, 0.0, e), zoom_level).x - scene_de(p - vec3<f32>(0.0, 0.0, e), zoom_level).x
    ));
}

fn calc_ao(pos: vec3<f32>, nor: vec3<f32>, zoom_level: f32) -> f32 {
    var occ: f32 = 0.0;
    var sca: f32 = 1.0;
    for (var i = 0; i < 5; i++) {
        let h = 0.01 + 0.08 * f32(i) / 4.0;
        let d = scene_de(pos + h * nor, zoom_level).x;
        occ += (h - d) * sca;
        sca *= 0.9;
    }
    return clamp(1.0 - 2.0 * occ, 0.0, 1.0);
}

fn ray_march(ro: vec3<f32>, rd: vec3<f32>, zoom_level: f32) -> vec3<f32> {
    var t: f32 = 0.0;
    var orbit_trap: f32 = 0.0;

    let surf_dist = 0.001;

    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let res = scene_de(p, zoom_level);
        let d = res.x;
        orbit_trap = res.y;

        if (abs(d) < surf_dist) {
            return vec3<f32>(t, orbit_trap, 1.0);
        }
        if (t > MAX_DIST) { break; }

        t += d * 0.5;  // Conservative stepping for detail
    }

    return vec3<f32>(-1.0, orbit_trap, 0.0);
}

fn star_field(rd: vec3<f32>) -> f32 {
    var p = rd * 300.0;
    var stars: f32 = 0.0;

    for (var i = 0; i < 3; i++) {
        let q = fract(p * (0.3 + f32(i) * 0.15)) - 0.5;
        let s = length(q);
        stars += smoothstep(0.05, 0.0, s) * (0.4 + f32(i) * 0.2);
        p = p * 1.4 + vec3<f32>(7.0, 11.0, 13.0);
    }

    return stars * 0.4;
}

fn background(rd: vec3<f32>) -> vec3<f32> {
    let stars = star_field(rd);

    var col = mix(
        vec3<f32>(0.02, 0.01, 0.03),
        vec3<f32>(0.05, 0.02, 0.08),
        rd.y * 0.5 + 0.5
    );

    col += stars * vec3<f32>(0.9, 0.95, 1.0);
    return col;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = (in.position.xy - 0.5 * uniforms.resolution) / uniforms.resolution.y;

    // Zoom level for infinite detail
    let zoom_level = uniforms.zoom_depth;
    let zoom_scale = exp(zoom_level);

    // Camera position scales with zoom - diving into the fractal
    let ro = uniforms.camera_pos / zoom_scale;
    let rd = normalize(
        uniforms.camera_forward +
        uv.x * uniforms.camera_right +
        uv.y * uniforms.camera_up
    );

    // Light follows camera
    let light_angle = uniforms.time * 0.3;
    let light_offset = vec3<f32>(sin(light_angle) * 2.0, 1.5, cos(light_angle) * 2.0) / zoom_scale;
    let light_pos = ro + light_offset;

    let hit = ray_march(ro, rd, zoom_level);

    var col: vec3<f32>;

    if (hit.z > 0.5) {
        let p = ro + rd * hit.x;
        let n = calc_normal(p, zoom_level);

        // Color varies with depth and orbit trap
        let trap = hit.y;
        var base_color = cosmic_palette(trap * 0.4 + zoom_level * 0.05, uniforms.color_shift);

        // Lighting
        let light_dir = normalize(light_pos - p);
        let view_dir = normalize(ro - p);
        let half_dir = normalize(light_dir + view_dir);

        let diff = max(dot(n, light_dir), 0.0);
        let spec = pow(max(dot(n, half_dir), 0.0), 32.0);
        let fresnel = pow(1.0 - max(dot(n, view_dir), 0.0), 3.0);
        let ao = calc_ao(p, n, zoom_level);

        let ambient = base_color * 0.12 * ao;
        let diffuse = base_color * diff * 0.7;
        let specular = vec3<f32>(1.0, 0.95, 0.9) * spec * 0.4;
        let rim = base_color * fresnel * 0.2;

        col = ambient + diffuse + specular + rim;

        // Depth fog
        let fog_amount = 1.0 - exp(-hit.x * zoom_scale * 0.15);
        col = mix(col, background(rd), fog_amount);

    } else {
        col = background(rd);
    }

    // Tone mapping
    col = col * (2.51 * col + 0.03) / (col * (2.43 * col + 0.59) + 0.14);

    // Gamma
    col = pow(col, vec3<f32>(1.0 / 2.2));

    // Vignette
    let vignette_uv = in.position.xy / uniforms.resolution;
    col *= 0.85 + 0.15 * pow(16.0 * vignette_uv.x * vignette_uv.y * (1.0 - vignette_uv.x) * (1.0 - vignette_uv.y), 0.25);

    return vec4<f32>(col, 1.0);
}

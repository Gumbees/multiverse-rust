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
    // Full screen triangle
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

const MAX_STEPS: i32 = 200;
const MAX_DIST: f32 = 50.0;
const SURF_DIST: f32 = 0.0005;
const PI: f32 = 3.14159265359;

// Cosmic color palette
fn cosmic_palette(t: f32, shift: f32) -> vec3<f32> {
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.0 + shift, 0.33 + shift, 0.67 + shift);
    return a + b * cos(6.28318 * (c * t + d));
}

// Mandelbulb distance estimator
fn mandelbulb_de(pos: vec3<f32>, power: f32) -> vec4<f32> {
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
            sin(new_phi) * sin(new_theta),
            cos(new_theta)
        );
        z += pos;

        orbit_trap = min(orbit_trap, length(z));
        orbit_trap = min(orbit_trap, abs(z.x) + abs(z.y) * 0.5);
    }

    return vec4<f32>(0.5 * log(r) * r / dr, orbit_trap, r, dr);
}

// Mandelbox distance estimator
fn mandelbox_de(pos: vec3<f32>, scale: f32) -> vec4<f32> {
    var z = pos;
    var dr: f32 = 1.0;
    var orbit_trap: f32 = 1e10;
    let fixed_radius2: f32 = 1.0;
    let min_radius2: f32 = 0.25;

    for (var i = 0; i < 12; i++) {
        // Box fold
        z = clamp(z, vec3<f32>(-1.0), vec3<f32>(1.0)) * 2.0 - z;

        // Sphere fold
        let r2 = dot(z, z);
        orbit_trap = min(orbit_trap, r2);

        if (r2 < min_radius2) {
            let temp = fixed_radius2 / min_radius2;
            z *= temp;
            dr *= temp;
        } else if (r2 < fixed_radius2) {
            let temp = fixed_radius2 / r2;
            z *= temp;
            dr *= temp;
        }

        z = scale * z + pos;
        dr = dr * abs(scale) + 1.0;
    }

    let r = length(z);
    return vec4<f32>(r / abs(dr), sqrt(orbit_trap), r, dr);
}

// Quaternion multiply
fn qmul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
        a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
        a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
        a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x
    );
}

// Quaternion Julia distance estimator
fn quaternion_julia_de(pos: vec3<f32>, w_param: f32) -> vec4<f32> {
    let c = vec4<f32>(-0.2, 0.6, 0.2, -0.4) * (w_param / 8.0);
    var z = vec4<f32>(pos, 0.0);
    var dz: f32 = 1.0;
    var orbit_trap: f32 = 1e10;

    for (var i = 0; i < 12; i++) {
        dz = 2.0 * length(z) * dz;
        z = qmul(z, z) + c;

        orbit_trap = min(orbit_trap, length(z.xyz));
        orbit_trap = min(orbit_trap, abs(z.x) + abs(z.y));

        if (dot(z, z) > 4.0) { break; }
    }

    let r = length(z);
    return vec4<f32>(0.5 * r * log(r) / dz, orbit_trap, r, dz);
}

// Main scene distance function
fn scene_sdf(p: vec3<f32>) -> vec4<f32> {
    // Add subtle variation based on position for "different realities"
    let reality_offset = sin(p.x * 0.3) * cos(p.y * 0.3) * sin(p.z * 0.3) * 0.02;
    let varied_power = uniforms.power + reality_offset * uniforms.power;

    if (uniforms.fractal_type == 0) {
        return mandelbulb_de(p, varied_power);
    } else if (uniforms.fractal_type == 1) {
        return mandelbox_de(p * 0.5, varied_power) * 2.0;
    } else {
        return quaternion_julia_de(p * 0.8, varied_power) * 1.25;
    }
}

// Calculate normal
fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let e = vec2<f32>(0.0005, 0.0);
    return normalize(vec3<f32>(
        scene_sdf(p + e.xyy).x - scene_sdf(p - e.xyy).x,
        scene_sdf(p + e.yxy).x - scene_sdf(p - e.yxy).x,
        scene_sdf(p + e.yyx).x - scene_sdf(p - e.yyx).x
    ));
}

// Ambient occlusion
fn calc_ao(pos: vec3<f32>, nor: vec3<f32>) -> f32 {
    var occ: f32 = 0.0;
    var sca: f32 = 1.0;
    for (var i = 0; i < 5; i++) {
        let h = 0.01 + 0.12 * f32(i) / 4.0;
        let d = scene_sdf(pos + h * nor).x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

// Soft shadows
fn soft_shadow(ro: vec3<f32>, rd: vec3<f32>, mint: f32, maxt: f32, k: f32) -> f32 {
    var res: f32 = 1.0;
    var t = mint;
    for (var i = 0; i < 24; i++) {
        if (t > maxt) { break; }
        let h = scene_sdf(ro + rd * t).x;
        res = min(res, k * h / t);
        if (res < 0.001) { break; }
        t += clamp(h, 0.01, 0.2);
    }
    return clamp(res, 0.0, 1.0);
}

// Ray march
fn ray_march(ro: vec3<f32>, rd: vec3<f32>) -> vec4<f32> {
    var t: f32 = 0.0;
    var orbit_trap: f32 = 0.0;
    var glow: f32 = 0.0;

    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let res = scene_sdf(p);
        let d = res.x;
        orbit_trap = res.y;

        // Accumulate glow for near-misses
        glow += 0.1 / (1.0 + d * d * 100.0);

        if (d < SURF_DIST) {
            return vec4<f32>(t, orbit_trap, glow, 1.0);
        }
        if (t > MAX_DIST) { break; }

        t += d * 0.7; // Slightly conservative step
    }

    return vec4<f32>(-1.0, orbit_trap, glow, 0.0);
}

// Star field background
fn star_field(rd: vec3<f32>) -> f32 {
    var p = rd * 500.0;
    var stars: f32 = 0.0;

    for (var i = 0; i < 3; i++) {
        let q = fract(p * (0.4 + f32(i) * 0.2)) - 0.5;
        let s = length(q);
        stars += smoothstep(0.1, 0.0, s) * (0.3 + f32(i) * 0.2);
        p = p * 1.5 + vec3<f32>(13.0, 17.0, 19.0);
    }

    return stars * 0.5;
}

// Background with nebula effect
fn background(rd: vec3<f32>, time: f32) -> vec3<f32> {
    let stars = star_field(rd);

    // Animated nebula
    let nebula_pos = rd * 2.0 + vec3<f32>(time * 0.02);
    let nebula = sin(nebula_pos.x * 3.0) * sin(nebula_pos.y * 3.0) * sin(nebula_pos.z * 3.0);
    let nebula_color = cosmic_palette(nebula * 0.5 + 0.5, uniforms.color_shift) * 0.15;

    // Base gradient
    var col = mix(
        vec3<f32>(0.01, 0.01, 0.03),
        vec3<f32>(0.05, 0.02, 0.08),
        rd.y * 0.5 + 0.5
    );

    col += nebula_color;
    col += stars * vec3<f32>(0.8, 0.9, 1.0);

    return col;
}

// Reality boundary effect - visual distortion at depth thresholds
fn reality_boundary(depth: f32, color: vec3<f32>) -> vec3<f32> {
    let boundary_freq = 3.0;
    let boundary = sin(depth * boundary_freq) * 0.5 + 0.5;
    let boundary_intensity = smoothstep(0.45, 0.5, boundary) * smoothstep(0.55, 0.5, boundary);

    // Chromatic shift at boundaries
    let shift = boundary_intensity * 0.3;
    return mix(color, cosmic_palette(depth * 0.1, uniforms.color_shift + shift), boundary_intensity * 0.5);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = (in.position.xy - 0.5 * uniforms.resolution) / uniforms.resolution.y;

    // Camera ray
    let ro = uniforms.camera_pos;
    let rd = normalize(
        uniforms.camera_forward +
        uv.x * uniforms.camera_right +
        uv.y * uniforms.camera_up
    );

    // Rotating light source
    let light_angle = uniforms.time * 0.3;
    let light_pos = ro + vec3<f32>(sin(light_angle) * 3.0, 2.0, cos(light_angle) * 3.0);

    let hit = ray_march(ro, rd);

    var col: vec3<f32>;

    if (hit.x > 0.0) {
        let p = ro + rd * hit.x;
        let n = calc_normal(p);

        // Orbit trap coloring with depth variation
        let trap = hit.y;
        let depth_color = uniforms.zoom_depth * 0.05 + length(p) * 0.02;
        var base_color = cosmic_palette(trap * 0.5 + 0.2 + depth_color, uniforms.color_shift);

        // Lighting
        let light_dir = normalize(light_pos - p);
        let view_dir = normalize(ro - p);
        let half_dir = normalize(light_dir + view_dir);

        // Diffuse
        let diff = max(dot(n, light_dir), 0.0);

        // Specular
        let spec = pow(max(dot(n, half_dir), 0.0), 32.0);

        // Fresnel
        let fresnel = pow(1.0 - max(dot(n, view_dir), 0.0), 3.0);

        // Ambient occlusion
        let ao = calc_ao(p, n);

        // Soft shadow
        let shadow = soft_shadow(p + n * 0.02, light_dir, 0.02, 5.0, 16.0);

        // Combine lighting
        let ambient = base_color * 0.15 * ao;
        let diffuse = base_color * diff * shadow * 0.7;
        let specular = vec3<f32>(0.8, 0.85, 1.0) * spec * shadow * 0.5;
        let rim = base_color * fresnel * 0.3;

        col = ambient + diffuse + specular + rim;

        // Add glow from near misses
        col += cosmic_palette(hit.z * 0.1, uniforms.color_shift) * hit.z * 0.02;

        // Depth fog
        let fog_amount = 1.0 - exp(-hit.x * 0.08);
        let fog_color = background(rd, uniforms.time);
        col = mix(col, fog_color, fog_amount);

        // Reality boundary effect
        col = reality_boundary(hit.x + uniforms.zoom_depth, col);

    } else {
        col = background(rd, uniforms.time);

        // Add glow even for misses (volumetric-ish effect)
        col += cosmic_palette(hit.z * 0.1 + uniforms.time * 0.1, uniforms.color_shift) * hit.z * 0.015;
    }

    // Tone mapping (ACES approximation)
    col = col * (2.51 * col + 0.03) / (col * (2.43 * col + 0.59) + 0.14);

    // Gamma correction
    col = pow(col, vec3<f32>(1.0 / 2.2));

    // Subtle vignette
    let vignette_uv = in.position.xy / uniforms.resolution;
    col *= 0.8 + 0.2 * pow(16.0 * vignette_uv.x * vignette_uv.y * (1.0 - vignette_uv.x) * (1.0 - vignette_uv.y), 0.2);

    return vec4<f32>(col, 1.0);
}

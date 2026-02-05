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

const MAX_STEPS: i32 = 200;
const MAX_DIST: f32 = 100.0;
const PI: f32 = 3.14159265359;

// ============================================================================
// CLEAN MANDELBULB - True fractal with infinite self-similar detail
// ============================================================================

fn mandelbulb_de(pos: vec3<f32>, power: f32) -> vec2<f32> {
    var z = pos;
    var dr: f32 = 1.0;
    var r: f32 = 0.0;
    var orbit_trap: f32 = 1e10;

    for (var i = 0; i < 15; i++) {
        r = length(z);
        if (r > 2.0) { break; }

        // Convert to spherical coordinates
        let theta = acos(z.z / r);
        let phi = atan2(z.y, z.x);

        // Derivative
        dr = pow(r, power - 1.0) * power * dr + 1.0;

        // Scale and rotate the point
        let zr = pow(r, power);
        let new_theta = theta * power;
        let new_phi = phi * power;

        // Convert back to cartesian
        z = zr * vec3<f32>(
            sin(new_theta) * cos(new_phi),
            sin(new_theta) * sin(new_phi),
            cos(new_theta)
        );
        z += pos;

        // Orbit trap for coloring
        orbit_trap = min(orbit_trap, length(z));
    }

    // Distance estimate
    let dist = 0.5 * log(r) * r / dr;
    return vec2<f32>(dist, orbit_trap);
}

// ============================================================================
// INFINITE ZOOM WRAPPER
// The key: scale coordinates to sample fractal at any depth
// ============================================================================

fn infinite_fractal_de(world_pos: vec3<f32>) -> vec2<f32> {
    // Direct fractal evaluation - camera already scaled by zoom
    let result = mandelbulb_de(world_pos, uniforms.power);
    return result;
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

fn calc_normal(p: vec3<f32>) -> vec3<f32> {
    let zoom = exp(uniforms.zoom_depth);
    let e = 0.0001 / zoom;
    return normalize(vec3<f32>(
        infinite_fractal_de(p + vec3<f32>(e, 0.0, 0.0)).x - infinite_fractal_de(p - vec3<f32>(e, 0.0, 0.0)).x,
        infinite_fractal_de(p + vec3<f32>(0.0, e, 0.0)).x - infinite_fractal_de(p - vec3<f32>(0.0, e, 0.0)).x,
        infinite_fractal_de(p + vec3<f32>(0.0, 0.0, e)).x - infinite_fractal_de(p - vec3<f32>(0.0, 0.0, e)).x
    ));
}

fn calc_ao(pos: vec3<f32>, nor: vec3<f32>) -> f32 {
    var occ: f32 = 0.0;
    var sca: f32 = 1.0;
    let zoom = exp(uniforms.zoom_depth);
    for (var i = 0; i < 5; i++) {
        let h = (0.01 + 0.1 * f32(i) / 4.0) / zoom;
        let d = infinite_fractal_de(pos + h * nor).x;
        occ += (h - d) * sca;
        sca *= 0.9;
    }
    return clamp(1.0 - 2.0 * occ, 0.0, 1.0);
}

fn ray_march(ro: vec3<f32>, rd: vec3<f32>) -> vec3<f32> {
    var t: f32 = 0.0;
    var orbit_trap: f32 = 0.0;

    // Surface and max distance scale with zoom
    let zoom = exp(uniforms.zoom_depth);
    let surf_dist = 0.0005 / zoom;
    let max_dist = MAX_DIST / zoom;

    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let res = infinite_fractal_de(p);
        let d = res.x;
        orbit_trap = res.y;

        if (d < surf_dist) {
            return vec3<f32>(t, orbit_trap, 1.0);
        }
        if (t > max_dist) { break; }

        t += d * 0.7;
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
        vec3<f32>(0.01, 0.01, 0.02),
        vec3<f32>(0.03, 0.01, 0.05),
        rd.y * 0.5 + 0.5
    );

    col += stars * vec3<f32>(0.9, 0.95, 1.0);
    return col;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let uv = (in.position.xy - 0.5 * uniforms.resolution) / uniforms.resolution.y;

    // Zoom factor
    let zoom = exp(uniforms.zoom_depth);

    // Camera setup - scale position by zoom to get closer in fractal space
    // This is the key: camera dives deeper as zoom increases
    let ro = uniforms.camera_pos / zoom;
    let rd = normalize(
        uniforms.camera_forward +
        uv.x * uniforms.camera_right +
        uv.y * uniforms.camera_up
    );

    // Rotating light follows camera, scaled by zoom
    let light_angle = uniforms.time * 0.3;
    let light_offset = vec3<f32>(sin(light_angle) * 2.0, 1.5, cos(light_angle) * 2.0) / zoom;
    let light_pos = ro + light_offset;

    let hit = ray_march(ro, rd);

    var col: vec3<f32>;

    if (hit.z > 0.5) {
        let p = ro + rd * hit.x;
        let n = calc_normal(p);

        // Color from orbit trap + zoom depth for variation
        let trap = hit.y;
        let depth_color = uniforms.zoom_depth * 0.1;
        var base_color = cosmic_palette(trap * 0.5 + depth_color, uniforms.color_shift);

        // Lighting
        let light_dir = normalize(light_pos - p);
        let view_dir = normalize(ro - p);
        let half_dir = normalize(light_dir + view_dir);

        let diff = max(dot(n, light_dir), 0.0);
        let spec = pow(max(dot(n, half_dir), 0.0), 32.0);
        let fresnel = pow(1.0 - max(dot(n, view_dir), 0.0), 3.0);
        let ao = calc_ao(p, n);

        let ambient = base_color * 0.15 * ao;
        let diffuse = base_color * diff * 0.7;
        let specular = vec3<f32>(1.0, 0.95, 0.9) * spec * 0.4;
        let rim = base_color * fresnel * 0.2;

        col = ambient + diffuse + specular + rim;

        // Depth fog - scale with zoom
        let fog_amount = 1.0 - exp(-hit.x * zoom * 0.1);
        col = mix(col, background(rd), fog_amount);

    } else {
        col = background(rd);
    }

    // Tone mapping (ACES)
    col = col * (2.51 * col + 0.03) / (col * (2.43 * col + 0.59) + 0.14);

    // Gamma
    col = pow(col, vec3<f32>(1.0 / 2.2));

    // Vignette
    let vignette_uv = in.position.xy / uniforms.resolution;
    col *= 0.85 + 0.15 * pow(16.0 * vignette_uv.x * vignette_uv.y * (1.0 - vignette_uv.x) * (1.0 - vignette_uv.y), 0.25);

    return vec4<f32>(col, 1.0);
}

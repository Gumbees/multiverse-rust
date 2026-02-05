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
const MAX_DIST: f32 = 100.0;
const SURF_DIST: f32 = 0.0001;
const PI: f32 = 3.14159265359;

// ============================================================================
// QUANTUM NOISE FUNCTIONS - Procedural coherent randomness
// ============================================================================

// Hash function for position-based seeding
fn hash31(p: vec3<f32>) -> f32 {
    var p3 = fract(p * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

// Smooth noise for coherent quantum states
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

// Fractal Brownian Motion - coherent detail at multiple scales
fn fbm(p: vec3<f32>, octaves: i32) -> f32 {
    var value: f32 = 0.0;
    var amplitude: f32 = 0.5;
    var frequency: f32 = 1.0;
    var pos = p;

    for (var i = 0; i < octaves; i++) {
        value += amplitude * noise3d(pos * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }

    return value;
}

// ============================================================================
// QUANTUM WAVEFUNCTION COLLAPSE
// The observation collapses infinite possibility into specific form
// ============================================================================

// Observation seed - deterministic for same viewpoint, changes when looking away
fn observation_seed(ray_origin: vec3<f32>, ray_dir: vec3<f32>, scale: f32) -> f32 {
    // Quantize the ray direction to create "measurement bins"
    // This means similar rays collapse to similar states (coherence)
    let quantized_dir = floor(ray_dir * 100.0) / 100.0;

    // The seed depends on where we're looking FROM and TO
    let origin_hash = hash31(ray_origin * scale);
    let dir_hash = hash31(quantized_dir * 1000.0 + vec3<f32>(origin_hash));

    // Time creates temporal decoherence - states drift
    let temporal_drift = sin(uniforms.time * 0.1) * 0.1;

    return dir_hash + temporal_drift;
}

// Collapse function - given a position, return the collapsed quantum state
// This is deterministic for the same position+scale but appears random
fn quantum_collapse(p: vec3<f32>, scale: f32, coherence: f32) -> f32 {
    // Multiple scale layers - infinite detail emerges from observation
    let base_scale = scale * 0.5;

    // Coherent noise gives smooth probability distribution
    let probability_field = fbm(p * base_scale, 5);

    // "Collapse" happens when we cross probability thresholds
    // The coherence parameter controls how sharp the collapse is
    let collapsed = smoothstep(0.3 - coherence * 0.2, 0.7 + coherence * 0.2, probability_field);

    return collapsed;
}

// ============================================================================
// QUANTUM FRACTAL DISTANCE ESTIMATOR
// The fractal itself is probabilistic - form emerges from observation
// ============================================================================

fn quantum_fractal_de(pos: vec3<f32>, obs_seed: f32, scale_depth: f32) -> vec4<f32> {
    var z = pos;
    var dr: f32 = 1.0;
    var r: f32 = 0.0;
    var orbit_trap: f32 = 1e10;

    // Scale determines how deep into infinite detail we go
    let effective_scale = exp(-scale_depth * 0.5);

    // Power fluctuates based on quantum state - different observations yield different fractals
    let power_base = uniforms.power;
    let power_fluctuation = (obs_seed - 0.5) * 2.0; // -1 to 1
    let power = power_base + power_fluctuation * 0.5;

    // Iterate the fractal with quantum perturbations
    for (var i = 0; i < 15; i++) {
        r = length(z);
        if (r > 2.0) { break; }

        // Convert to polar
        var theta = acos(z.z / r);
        var phi = atan2(z.y, z.x);

        // Quantum perturbation at each iteration
        // This creates different detail at each observation
        let iter_seed = hash31(z * effective_scale + vec3<f32>(f32(i) * 0.1));
        let quantum_perturb = (iter_seed - 0.5) * 0.1 * (1.0 - effective_scale);

        theta += quantum_perturb;
        phi += quantum_perturb * 0.5;

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

        // Orbit trap for coloring - tracks the "probability density"
        orbit_trap = min(orbit_trap, length(z));
        orbit_trap = min(orbit_trap, abs(z.x) + abs(z.y) * 0.5);
    }

    // Distance estimate with scale factor for infinite zoom
    let dist = 0.5 * log(r) * r / dr * effective_scale;

    return vec4<f32>(dist, orbit_trap, r, obs_seed);
}

// Multi-scale quantum fractal - detail emerges at each scale layer
fn multi_scale_quantum_de(pos: vec3<f32>, obs_seed: f32) -> vec4<f32> {
    // Calculate the current scale based on camera distance from origin
    let cam_dist = length(uniforms.camera_pos);
    let scale_depth = log(max(cam_dist, 0.001)) * -1.0 + uniforms.zoom_depth;

    // Get base fractal
    var result = quantum_fractal_de(pos, obs_seed, scale_depth);

    // Add detail layers that emerge at deeper scales
    let num_detail_layers = i32(min(scale_depth * 2.0, 5.0));

    for (var layer = 1; layer <= 5; layer++) {
        if (layer > num_detail_layers) { break; }

        let layer_scale = pow(2.0, f32(layer));
        let layer_seed = hash31(pos * layer_scale + vec3<f32>(obs_seed * f32(layer)));

        // Each layer adds finer detail
        let detail = quantum_collapse(pos * layer_scale, layer_scale, 0.5);

        // Modulate the distance with detail (creates new structure at each scale)
        result.x *= 1.0 + (detail - 0.5) * 0.3 / layer_scale;
    }

    return result;
}

// ============================================================================
// LIGHTING AND RENDERING
// ============================================================================

fn cosmic_palette(t: f32, shift: f32) -> vec3<f32> {
    let a = vec3<f32>(0.5, 0.5, 0.5);
    let b = vec3<f32>(0.5, 0.5, 0.5);
    let c = vec3<f32>(1.0, 1.0, 1.0);
    let d = vec3<f32>(0.0 + shift, 0.33 + shift, 0.67 + shift);
    return a + b * cos(6.28318 * (c * t + d));
}

fn calc_normal(p: vec3<f32>, obs_seed: f32) -> vec3<f32> {
    let e = vec2<f32>(0.0001, 0.0);
    return normalize(vec3<f32>(
        multi_scale_quantum_de(p + e.xyy, obs_seed).x - multi_scale_quantum_de(p - e.xyy, obs_seed).x,
        multi_scale_quantum_de(p + e.yxy, obs_seed).x - multi_scale_quantum_de(p - e.yxy, obs_seed).x,
        multi_scale_quantum_de(p + e.yyx, obs_seed).x - multi_scale_quantum_de(p - e.yyx, obs_seed).x
    ));
}

fn calc_ao(pos: vec3<f32>, nor: vec3<f32>, obs_seed: f32) -> f32 {
    var occ: f32 = 0.0;
    var sca: f32 = 1.0;
    for (var i = 0; i < 5; i++) {
        let h = 0.01 + 0.08 * f32(i) / 4.0;
        let d = multi_scale_quantum_de(pos + h * nor, obs_seed).x;
        occ += (h - d) * sca;
        sca *= 0.95;
    }
    return clamp(1.0 - 3.0 * occ, 0.0, 1.0);
}

fn ray_march(ro: vec3<f32>, rd: vec3<f32>, obs_seed: f32) -> vec4<f32> {
    var t: f32 = 0.0;
    var orbit_trap: f32 = 0.0;
    var glow: f32 = 0.0;
    var quantum_state: f32 = 0.0;

    // Adaptive step size based on scale
    let scale_factor = exp(-uniforms.zoom_depth * 0.3);
    let min_dist = SURF_DIST * scale_factor;

    for (var i = 0; i < MAX_STEPS; i++) {
        let p = ro + rd * t;
        let res = multi_scale_quantum_de(p, obs_seed);
        let d = res.x;
        orbit_trap = res.y;
        quantum_state = res.w;

        // Accumulate glow from probability density
        glow += 0.1 / (1.0 + d * d * 100.0);

        if (d < min_dist) {
            return vec4<f32>(t, orbit_trap, glow, quantum_state);
        }
        if (t > MAX_DIST * scale_factor) { break; }

        // Adaptive stepping
        t += d * 0.6;
    }

    return vec4<f32>(-1.0, orbit_trap, glow, quantum_state);
}

fn star_field(rd: vec3<f32>) -> f32 {
    var p = rd * 500.0;
    var stars: f32 = 0.0;

    for (var i = 0; i < 3; i++) {
        let q = fract(p * (0.4 + f32(i) * 0.2)) - 0.5;
        let s = length(q);
        stars += smoothstep(0.08, 0.0, s) * (0.3 + f32(i) * 0.2);
        p = p * 1.5 + vec3<f32>(13.0, 17.0, 19.0);
    }

    return stars * 0.5;
}

fn background(rd: vec3<f32>, time: f32) -> vec3<f32> {
    let stars = star_field(rd);

    // Quantum foam background - probability waves
    let foam_pos = rd * 5.0 + vec3<f32>(time * 0.05);
    let foam = fbm(foam_pos, 3) * 0.15;

    var col = mix(
        vec3<f32>(0.01, 0.01, 0.02),
        vec3<f32>(0.04, 0.02, 0.06),
        rd.y * 0.5 + 0.5
    );

    col += cosmic_palette(foam + time * 0.02, uniforms.color_shift) * foam * 0.5;
    col += stars * vec3<f32>(0.8, 0.9, 1.0);

    return col;
}

// Probability field visualization - shows the quantum superposition
fn probability_overlay(p: vec3<f32>, normal_col: vec3<f32>) -> vec3<f32> {
    let prob = quantum_collapse(p, 1.0, 0.3);
    let prob_color = cosmic_palette(prob, uniforms.color_shift + 0.3);

    // Subtle probability visualization
    return mix(normal_col, prob_color, 0.1);
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

    // Calculate observation seed - THIS IS THE MEASUREMENT
    // Same position + direction = same collapsed state (coherence)
    // Different observation = possibility of different state
    let obs_seed = observation_seed(ro, rd, 1.0);

    // Rotating light
    let light_angle = uniforms.time * 0.2;
    let light_pos = ro + vec3<f32>(sin(light_angle) * 3.0, 2.0, cos(light_angle) * 3.0);

    let hit = ray_march(ro, rd, obs_seed);

    var col: vec3<f32>;

    if (hit.x > 0.0) {
        let p = ro + rd * hit.x;
        let n = calc_normal(p, obs_seed);

        // Color based on orbit trap and quantum state
        let trap = hit.y;
        let q_state = hit.w;
        var base_color = cosmic_palette(trap * 0.4 + q_state * 0.3 + 0.2, uniforms.color_shift);

        // Add probability visualization
        base_color = probability_overlay(p, base_color);

        // Lighting
        let light_dir = normalize(light_pos - p);
        let view_dir = normalize(ro - p);
        let half_dir = normalize(light_dir + view_dir);

        let diff = max(dot(n, light_dir), 0.0);
        let spec = pow(max(dot(n, half_dir), 0.0), 32.0);
        let fresnel = pow(1.0 - max(dot(n, view_dir), 0.0), 3.0);
        let ao = calc_ao(p, n, obs_seed);

        let ambient = base_color * 0.12 * ao;
        let diffuse = base_color * diff * 0.7;
        let specular = vec3<f32>(0.8, 0.85, 1.0) * spec * 0.5;
        let rim = base_color * fresnel * 0.25;

        col = ambient + diffuse + specular + rim;

        // Glow from probability density
        col += cosmic_palette(hit.z * 0.1, uniforms.color_shift) * hit.z * 0.015;

        // Depth fog
        let scale_factor = exp(-uniforms.zoom_depth * 0.3);
        let fog_amount = 1.0 - exp(-hit.x * 0.05 / scale_factor);
        col = mix(col, background(rd, uniforms.time), fog_amount);

    } else {
        col = background(rd, uniforms.time);

        // Even misses show the probability glow
        col += cosmic_palette(hit.z * 0.1 + uniforms.time * 0.05, uniforms.color_shift) * hit.z * 0.01;
    }

    // Tone mapping
    col = col * (2.51 * col + 0.03) / (col * (2.43 * col + 0.59) + 0.14);

    // Gamma
    col = pow(col, vec3<f32>(1.0 / 2.2));

    // Vignette
    let vignette_uv = in.position.xy / uniforms.resolution;
    col *= 0.8 + 0.2 * pow(16.0 * vignette_uv.x * vignette_uv.y * (1.0 - vignette_uv.x) * (1.0 - vignette_uv.y), 0.2);

    return vec4<f32>(col, 1.0);
}

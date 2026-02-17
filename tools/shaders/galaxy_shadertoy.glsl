// Fork of "GAMER Galaxy viewer " by michael0884. https://shadertoy.com/view/W3Gfz1
// 2026-02-17 02:17:02

// preset constants
#define PRESET_SPIRAL 0
#define PRESET_SOMBRERO 1
#define PRESET_SB0 2
#define PRESET_IRREGULAR 3
#define PRESET_IRREGULAR2 4
#define PRESET_REDBAR 5
#define PRESET_TONSOFARMS 6
#define PRESET_VORTEXCLOUD 7
#define PRESET_WHEELGALAXY 8
#define GALAXY_PRESET PRESET_WHEELGALAXY // edit to choose preset

// render tuning
#define MARCH_MAX_STEPS 512
#define MARCH_BASE_STEP 0.005
#define MARCH_STEP_ALPHA 0.002
#define MARCH_DENSITY_STEP_SCALE 200.0
#define MARCH_WEIGHT_THRESHOLD 0.0005
#define MARCH_SAMPLE_DITHER 1.0
#define DISK_EDGE_FADE_START 0.82
#define DISK_EDGE_FADE_END 1.02
#define VOLUME_EDGE_FADE_START 0.90
#define VOLUME_EDGE_FADE_END 1.02
#define HEIGHT_EDGE_FADE_START 1.6
#define HEIGHT_EDGE_FADE_END 2.2

// post / tonemap
#define POST_TANH_STRENGTH 0.04

// camera / view
#define CAMERA_RADIUS 0.4
#define CAMERA_FOV_DEG 60.0
#define CAMERA_IDLE_AZ_SPEED 0.35
#define CAMERA_IDLE_EL 0.6
#define CAMERA_MOUSE_EL_BIAS 0.6
#define CAMERA_EL_CLAMP 1.45

const float PI = 3.14159265;
const int MAX_COMPONENTS = 8;

struct GalaxyParams {
    vec3 axis;
    float winding_b;
    float winding_n;
    float no_arms;
    vec4 arms;
};

struct MarchParams {
    vec3 camera;
    vec3 dir;
    vec2 fragCoord;
    float t0;
    float tFar;
    float step;
    float rayStep;
    GalaxyParams g;
};

struct SampleContribution {
    vec3 emit;
    vec3 trans;
};

struct ComponentParams {
    int cid;
    float strength;
    float arm;
    float z0;
    float r0;
    float isActive;
    float delta;
    float winding_mul;
    float scale;
    float noise_offset;
    float noise_tilt;
    float ks;
    float inner;
    vec3 spec;
};

const ivec3 grad3[16] = ivec3[16](
    ivec3(1,1,0), ivec3(-1,1,0), ivec3(1,-1,0), ivec3(-1,-1,0),
    ivec3(1,0,1), ivec3(-1,0,1), ivec3(1,0,-1), ivec3(-1,0,-1),
    ivec3(0,1,1), ivec3(0,-1,1), ivec3(0,1,-1), ivec3(0,-1,-1),
    ivec3(1,1,0), ivec3(-1,1,0), ivec3(1,-1,0), ivec3(-1,-1,0)
);

int fastfloor(float x) { return x > 0.0 ? int(x) : int(x) - 1; }
float dot3i(ivec3 g, float x, float y, float z) { return float(g.x)*x + float(g.y)*y + float(g.z)*z; }
float pseudo_blue_noise(vec2 p)
{
    vec2 px = floor(p);
    return fract(52.9829189 * fract(dot(px, vec2(0.06711056, 0.00583715))));
}
uint hash_u32(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }
uint hash_i3(ivec3 p)
{
    uint h = uint(p.x) * 0x8da6b343u;
    h ^= uint(p.y) * 0xd8163841u;
    h ^= uint(p.z) * 0xcb1ab31fu;
    return hash_u32(h);
}

float raw_noise_3d(float x, float y, float z)
{
    float n0, n1, n2, n3;
    float F3 = 1.0/3.0;
    float s = (x+y+z)*F3;
    int i = fastfloor(x+s);
    int j = fastfloor(y+s);
    int k = fastfloor(z+s);

    float G3 = 1.0/6.0;
    float t = float(i+j+k)*G3;
    float X0 = float(i)-t;
    float Y0 = float(j)-t;
    float Z0 = float(k)-t;
    float x0 = x-X0;
    float y0 = y-Y0;
    float z0 = z-Z0;

    int i1, j1, k1;
    int i2, j2, k2;

    if (x0 >= y0) {
        if (y0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=1; k2=0; }
        else if (x0 >= z0) { i1=1; j1=0; k1=0; i2=1; j2=0; k2=1; }
        else { i1=0; j1=0; k1=1; i2=1; j2=0; k2=1; }
    } else {
        if (y0 < z0) { i1=0; j1=0; k1=1; i2=0; j2=1; k2=1; }
        else if (x0 < z0) { i1=0; j1=1; k1=0; i2=0; j2=1; k2=1; }
        else { i1=0; j1=1; k1=0; i2=1; j2=1; k2=0; }
    }

    float x1 = x0-float(i1)+G3;
    float y1 = y0-float(j1)+G3;
    float z1 = z0-float(k1)+G3;
    float x2 = x0-float(i2)+2.0*G3;
    float y2 = y0-float(j2)+2.0*G3;
    float z2 = z0-float(k2)+2.0*G3;
    float x3 = x0-1.0+3.0*G3;
    float y3 = y0-1.0+3.0*G3;
    float z3 = z0-1.0+3.0*G3;

    int gi0 = int(hash_i3(ivec3(i, j, k)) & 15u);
    int gi1 = int(hash_i3(ivec3(i + i1, j + j1, k + k1)) & 15u);
    int gi2 = int(hash_i3(ivec3(i + i2, j + j2, k + k2)) & 15u);
    int gi3 = int(hash_i3(ivec3(i + 1, j + 1, k + 1)) & 15u);

    float t0 = 0.6 - x0*x0 - y0*y0 - z0*z0;
    if (t0 < 0.0) n0 = 0.0;
    else { t0 *= t0; n0 = t0*t0*dot3i(grad3[gi0], x0, y0, z0); }

    float t1 = 0.6 - x1*x1 - y1*y1 - z1*z1;
    if (t1 < 0.0) n1 = 0.0;
    else { t1 *= t1; n1 = t1*t1*dot3i(grad3[gi1], x1, y1, z1); }

    float t2 = 0.6 - x2*x2 - y2*y2 - z2*z2;
    if (t2 < 0.0) n2 = 0.0;
    else { t2 *= t2; n2 = t2*t2*dot3i(grad3[gi2], x2, y2, z2); }

    float t3 = 0.6 - x3*x3 - y3*y3 - z3*z3;
    if (t3 < 0.0) n3 = 0.0;
    else { t3 *= t3; n3 = t3*t3*dot3i(grad3[gi3], x3, y3, z3); }

    return 32.0 * (n0+n1+n2+n3);
}

float octave_noise_3d(float octaves, float persistence, float scale, vec3 p)
{
    float total = 0.0;
    float frequency = scale;
    float amplitude = 1.0;
    float maxAmplitude = 0.0;
    for (int i=0; i<int(octaves); ++i) {
        total += raw_noise_3d(p.x*frequency, p.y*frequency, p.z*frequency) * amplitude;
        frequency *= 2.0;
        maxAmplitude += amplitude;
        amplitude *= persistence;
    }
    return total / maxAmplitude;
}

float get_ridged_mf(vec3 p, float frequency, int octaves, float lacunarity, float offset, float gain)
{
    float value = 0.0;
    float weight = 1.0;
    float w = -0.05;
    vec3 vt = p;
    float freq = frequency;
    for (int octave=0; octave<octaves; ++octave) {
        float signal = raw_noise_3d(vt.x, vt.y, vt.z);
        signal = abs(signal);
        signal = offset - signal;
        signal *= signal;
        signal *= weight;
        weight = clamp(signal * gain, 0.0, 1.0);
        value += signal * pow(freq, w);
        vt *= lacunarity;
        freq *= lacunarity;
    }
    return (value * 1.25) - 1.0;
}

vec3 spectrum_from_id(int sid)
{
    if (sid == 0) return vec3(1.0, 0.6, 0.4);  // red
    if (sid == 1) return vec3(1.0, 0.9, 0.45); // yellow
    if (sid == 2) return vec3(0.4, 0.6, 1.0);  // blue
    if (sid == 3) return vec3(1.0, 1.0, 1.0);  // white
    if (sid == 4) return vec3(0.3, 0.7, 1.0);  // cyan
    return vec3(1.0, 0.3, 0.8);                // purple
}

ComponentParams make_component(
    int cid,
    float strength, float arm, float z0, float r0,
    float isActive, float delta, float winding_mul, float scale,
    float noise_offset, float noise_tilt, float ks, float inner,
    vec3 spec)
{
    ComponentParams c;
    c.cid = cid;
    c.strength = strength;
    c.arm = arm;
    c.z0 = z0;
    c.r0 = r0;
    c.isActive = isActive;
    c.delta = delta;
    c.winding_mul = winding_mul;
    c.scale = scale;
    c.noise_offset = noise_offset;
    c.noise_tilt = noise_tilt;
    c.ks = ks;
    c.inner = inner;
    c.spec = spec;
    return c;
}

void get_galaxy_params(out vec3 axis, out float winding_b, out float winding_n, out float no_arms, out vec4 arms)
{
#if GALAXY_PRESET == PRESET_SPIRAL
    axis = vec3(1.0, 1.0, 1.0);
    winding_b = 0.5; winding_n = 4.0; no_arms = 2.0;
    arms = vec4(0.0, 3.14, 1.57, 4.71);
#elif GALAXY_PRESET == PRESET_SOMBRERO
    axis = vec3(1.0, 1.0, 1.0);
    winding_b = 0.4; winding_n = 4.0; no_arms = 2.0;
    arms = vec4(0.0, 3.14, 1.57, 4.71);
#elif GALAXY_PRESET == PRESET_SB0
    axis = vec3(1.0, 1.0, 1.0);
    winding_b = 1.0; winding_n = 4.0; no_arms = 2.0;
    arms = vec4(0.0, 3.14, 1.57, 4.71);
#elif GALAXY_PRESET == PRESET_IRREGULAR
    axis = vec3(1.0, 1.0, 1.0);
    winding_b = 1.0; winding_n = 2.0; no_arms = 1.0;
    arms = vec4(0.0, 3.14, 1.57, 4.71);
#elif GALAXY_PRESET == PRESET_IRREGULAR2
    axis = vec3(1.0, 1.0, 1.0);
    winding_b = 2.0; winding_n = 2.0; no_arms = 3.0;
    arms = vec4(0.0, 3.14, 2.5, 2.71);
#elif GALAXY_PRESET == PRESET_REDBAR
    axis = vec3(1.0, 1.0, 1.0);
    winding_b = 3.0; winding_n = 4.0; no_arms = 2.0;
    arms = vec4(0.0, 3.14, 1.57, 4.71);
#elif GALAXY_PRESET == PRESET_TONSOFARMS
    axis = vec3(1.0, 1.0, 1.0);
    winding_b = 0.5; winding_n = 5.0; no_arms = 4.0;
    arms = vec4(0.0, 3.14, 1.57, 4.71);
#elif GALAXY_PRESET == PRESET_VORTEXCLOUD
    axis = vec3(1.0, 1.0, 1.0);
    winding_b = 0.54; winding_n = 4.0; no_arms = 2.0;
    arms = vec4(0.0, 3.14, 1.57, 4.71);
#elif GALAXY_PRESET == PRESET_WHEELGALAXY
    axis = vec3(1.0, 1.0, 1.0);
    winding_b = 1.0; winding_n = 4.0; no_arms = 2.0;
    arms = vec4(0.0, 3.14, 1.57, 4.71);
#else
    axis = vec3(1.0, 1.0, 1.0);
    winding_b = 0.5; winding_n = 4.0; no_arms = 2.0;
    arms = vec4(0.0, 3.14, 1.57, 4.71);
#endif
}

int component_count()
{
#if GALAXY_PRESET == PRESET_SPIRAL
    return 6;
#elif GALAXY_PRESET == PRESET_SOMBRERO
    return 6;
#elif GALAXY_PRESET == PRESET_SB0
    return 6;
#elif GALAXY_PRESET == PRESET_IRREGULAR
    return 7;
#elif GALAXY_PRESET == PRESET_IRREGULAR2
    return 8;
#elif GALAXY_PRESET == PRESET_REDBAR
    return 6;
#elif GALAXY_PRESET == PRESET_TONSOFARMS
    return 6;
#elif GALAXY_PRESET == PRESET_VORTEXCLOUD
    return 3;
#elif GALAXY_PRESET == PRESET_WHEELGALAXY
    return 6;
#else
    return 0;
#endif
}

void load_component(int i, out ComponentParams c)
{
#if GALAXY_PRESET == PRESET_SPIRAL
    if (i==0) { c = make_component(0, 25.0, 1.0, 0.02, 4.0, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.1, spectrum_from_id(1)); return; }
    if (i==1) { c = make_component(1, 5000.0, 0.3, 0.02, 0.4, 1.0, 0.0, 0.1, 10.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==2) { c = make_component(3, 150.0, 0.2, 0.03, 0.45, 1.0, 0.0, 0.07, 5.0, 1.0, 1.5, 1.0, 0.2, spectrum_from_id(2)); return; }
    if (i==3) { c = make_component(0, 30.0, 1.0, 0.02, 10.0, 0.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(4)); return; }
    if (i==4) { c = make_component(5, 10.0, 0.1, 0.05, 0.3, 1.0, 0.0, 0.0, 6.0, 0.0, 15.0, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==5) { c = make_component(1, 1000.0, 0.02, 0.02, 0.3, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(0)); return; }
    c = make_component(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec3(0.0)); return;
#elif GALAXY_PRESET == PRESET_SOMBRERO
    if (i==0) { c = make_component(0, 30.0, 1.0, 0.02, 8.0, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.1, spectrum_from_id(1)); return; }
    if (i==1) { c = make_component(1, 400.0, 0.01, 0.04, 0.5, 1.0, 0.0, 0.1, 1.0, 0.0, 0.01, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==2) { c = make_component(3, 50.0, 0.1, 0.02, 0.4, 1.0, 0.0, 0.1, 4.0, 1.2, 1.5, 1.0, 0.6, spectrum_from_id(2)); return; }
    if (i==3) { c = make_component(3, 30.0, 0.5, 0.02, 0.15, 1.0, 0.0, 0.1, 5.0, 1.1, 1.5, 1.0, 0.05, spectrum_from_id(2)); return; }
    if (i==4) { c = make_component(5, 3.0, 0.1, 0.05, 0.5, 1.0, 0.0, 0.1, 150.0, 0.0, 12.0, 1.0, 0.0, spectrum_from_id(0)); return; }
    if (i==5) { c = make_component(0, 5.0, 0.1, 0.05, 4.0, 1.0, 0.0, 0.1, 150.0, 0.0, 12.0, 1.0, 0.0, spectrum_from_id(0)); return; }
    c = make_component(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec3(0.0)); return;
#elif GALAXY_PRESET == PRESET_SB0
    if (i==0) { c = make_component(0, 250.0, 1.0, 0.02, 25.0, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.1, spectrum_from_id(1)); return; }
    if (i==1) { c = make_component(1, 5000.0, 0.4, 0.02, 0.3, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==2) { c = make_component(3, 500.0, 0.9, 0.015, 0.35, 1.0, 0.0, 0.1, 20.0, 1.0, 1.2, 1.0, 0.02, spectrum_from_id(2)); return; }
    if (i==3) { c = make_component(0, 30.0, 1.0, 0.02, 10.0, 0.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(4)); return; }
    if (i==4) { c = make_component(5, 15.0, 0.5, 0.03, 0.3, 1.0, 0.0, 0.0, 150.0, 0.0, 15.0, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==5) { c = make_component(1, 1000.0, 0.02, 0.02, 0.3, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(0)); return; }
    c = make_component(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec3(0.0)); return;
#elif GALAXY_PRESET == PRESET_IRREGULAR
    if (i==0) { c = make_component(1, 1500.0, 0.1, 0.02, 0.5, 1.0, 1.5, 0.1, 1.0, 0.0, 0.2, 1.0, 0.2, spectrum_from_id(2)); return; }
    if (i==1) { c = make_component(1, 500.0, 0.1, 0.05, 0.4, 1.0, 1.5, 0.1, 1.0, 0.0, 0.3, 1.0, 0.0, spectrum_from_id(0)); return; }
    if (i==2) { c = make_component(3, 10.0, 0.05, 0.025, 0.45, 1.0, 0.0, 0.05, 2.0, 1.5, 1.5, 1.0, 0.3, spectrum_from_id(2)); return; }
    if (i==3) { c = make_component(5, 5.0, 0.05, 0.02, 0.4, 1.0, 0.5, 0.1, 5.0, 0.0, 15.0, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==4) { c = make_component(0, 60.0, 1.0, 0.02, 10.0, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(1)); return; }
    if (i==5) { c = make_component(5, 5.0, 0.01, 0.07, 0.6, 1.0, 0.0, 0.1, 100.0, 0.0, 8.0, 1.0, 0.1, spectrum_from_id(0)); return; }
    if (i==6) { c = make_component(3, 50.0, 0.15, 0.02, 0.3, 1.0, 2.5, 0.35, 7.0, 1.0, 1.5, 1.0, 0.01, spectrum_from_id(2)); return; }
    c = make_component(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec3(0.0)); return;
#elif GALAXY_PRESET == PRESET_IRREGULAR2
    if (i==0) { c = make_component(1, 1500.0, 0.1, 0.02, 0.5, 1.0, 1.5, 0.1, 1.0, 0.0, 0.2, 1.0, 0.2, spectrum_from_id(2)); return; }
    if (i==1) { c = make_component(1, 500.0, 0.4, 0.05, 0.4, 1.0, 1.5, 0.1, 1.0, 0.0, 0.3, 1.0, 0.25, spectrum_from_id(0)); return; }
    if (i==2) { c = make_component(3, 20.0, 0.2, 0.025, 0.45, 1.0, 0.0, 0.5, 5.0, 1.0, 1.5, 1.0, 0.2, spectrum_from_id(2)); return; }
    if (i==3) { c = make_component(5, 5.0, 0.1, 0.02, 0.4, 1.0, 0.5, 0.1, 5.0, 0.0, 15.0, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==4) { c = make_component(0, 250.0, 0.1, 0.02, 15.0, 1.0, 1.5, 0.1, 1.0, 0.0, 0.2, 1.0, 0.2, spectrum_from_id(3)); return; }
    if (i==5) { c = make_component(5, 5.0, 0.01, 0.07, 0.6, 1.0, 0.0, 0.1, 100.0, 0.0, 8.0, 1.0, 0.1, spectrum_from_id(0)); return; }
    if (i==6) { c = make_component(3, 25.0, 0.35, 0.02, 0.3, 1.0, 2.5, 0.35, 5.0, 1.1, 1.5, 1.0, 0.01, spectrum_from_id(2)); return; }
    if (i==7) { c = make_component(5, 15.0, 0.1, 0.04, 0.5, 1.0, 1.0, 0.2, 10.0, 0.0, 10.0, 1.0, 0.2, spectrum_from_id(1)); return; }
    c = make_component(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec3(0.0)); return;
#elif GALAXY_PRESET == PRESET_REDBAR
    if (i==0) { c = make_component(0, 13.0, 1.0, 0.03, 3.0, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.1, spectrum_from_id(1)); return; }
    if (i==1) { c = make_component(1, 500.0, 0.1, 0.03, 0.45, 1.0, 0.0, 0.1, 5.0, 0.0, 0.5, -0.5, 0.0, spectrum_from_id(0)); return; }
    if (i==2) { c = make_component(0, 250.0, 1.0, 0.02, 15.0, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==3) { c = make_component(2, 150.0, 0.2, 0.01, 0.45, 1.0, 0.0, 0.2, 1.0, 0.0, 1.0, -0.8, 0.0, spectrum_from_id(2)); return; }
    if (i==4) { c = make_component(5, 50.0, 0.3, 0.04, 0.4, 1.0, 0.0, 0.1, 30.0, 0.0, 10.0, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==5) { c = make_component(3, 300.0, 2.0, 0.02, 5.0, 1.0, 0.0, 0.1, 8.0, 1.0, 1.5, 1.0, 0.0, spectrum_from_id(2)); return; }
    c = make_component(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec3(0.0)); return;
#elif GALAXY_PRESET == PRESET_TONSOFARMS
    if (i==0) { c = make_component(0, 100.0, 1.0, 0.03, 20.1, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.1, spectrum_from_id(3)); return; }
    if (i==1) { c = make_component(1, 500.0, 0.05, 0.03, 0.45, 1.0, 0.0, 0.1, 5.0, 0.0, 0.5, 1.0, 0.0, spectrum_from_id(0)); return; }
    if (i==2) { c = make_component(5, 5.0, 0.3, 0.04, 0.4, 1.0, 0.0, 0.1, 50.0, 0.0, 15.0, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==3) { c = make_component(0, 20.0, 1.0, 0.02, 5.0, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(1)); return; }
    if (i==4) { c = make_component(1, 3900.0, 0.3, 0.02, 0.4, 1.0, 0.0, 0.1, 3.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(2)); return; }
    if (i==5) { c = make_component(3, 150.0, 0.5, 0.015, 0.4, 1.0, 0.0, 0.1, 6.0, 1.05, 1.5, 1.0, 0.25, spectrum_from_id(2)); return; }
    c = make_component(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec3(0.0)); return;
#elif GALAXY_PRESET == PRESET_VORTEXCLOUD
    if (i==0) { c = make_component(0, 30.0, 1.0, 0.02, 3.0, 1.0, 0.0, 0.5, 1.0, 0.0, 1.0, 1.0, 0.1, spectrum_from_id(4)); return; }
    if (i==1) { c = make_component(3, 15.0, 0.1, 0.05, 0.5, 1.0, 0.0, 0.2, 2.0, 1.3, 1.0, 1.0, 0.25, spectrum_from_id(2)); return; }
    if (i==2) { c = make_component(1, 100.0, 0.01, 0.03, 0.6, 1.0, 0.0, 0.1, 1.0, 0.0, 0.01, 1.0, 0.0, spectrum_from_id(0)); return; }
    c = make_component(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec3(0.0)); return;
#elif GALAXY_PRESET == PRESET_WHEELGALAXY
    if (i==0) { c = make_component(0, 30.0, 1.0, 0.02, 5.0, 1.0, 0.0, 0.1, 1.0, 0.0, 1.0, 1.0, 0.0, spectrum_from_id(1)); return; }
    if (i==1) { c = make_component(1, 900.0, 0.01, 0.03, 0.4, 1.0, 0.0, 0.1, 1.0, 0.0, 0.4, 1.0, 0.5, spectrum_from_id(2)); return; }
    if (i==2) { c = make_component(3, 250.0, 0.25, 0.02, 0.4, 1.0, 0.0, 0.1, 10.0, 1.0, 1.1, 1.0, 0.6, spectrum_from_id(2)); return; }
    if (i==3) { c = make_component(5, 0.5, 0.05, 0.02, 0.5, 1.0, 0.0, 0.1, 6.0, 0.0, 20.0, 1.0, 0.3, spectrum_from_id(2)); return; }
    if (i==4) { c = make_component(5, 100.0, 0.1, 0.02, 0.55, 1.0, 0.0, 0.1, 100.0, 0.0, 10.0, 1.0, 0.3, spectrum_from_id(0)); return; }
    if (i==5) { c = make_component(3, 50.0, 0.2, 0.02, 0.2, 1.0, 0.0, 0.25, 5.0, 1.1, 1.0, 1.0, 0.05, spectrum_from_id(2)); return; }
    c = make_component(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec3(0.0)); return;
#else
    c = make_component(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vec3(0.0)); return;
#endif
}

bool intersect_sphere(vec3 o, vec3 d, vec3 axis, out vec3 isp1, out vec3 isp2, out float t0, out float t1)
{
    vec3 inv = vec3(1.0/(axis.x*axis.x), 1.0/(axis.y*axis.y), 1.0/(axis.z*axis.z));
    vec3 rD = d * inv;
    vec3 rO = o * inv;
    float A = dot(d, rD);
    float B = 2.0 * dot(d, rO);
    float C = dot(o, rO) - 1.0;
    float S = B*B - 4.0*A*C;
    if (S <= 0.0) { isp1=vec3(0); isp2=vec3(0); t0=0.0; t1=0.0; return false; }
    t0 = (-B - sqrt(S)) / (2.0*A);
    t1 = (-B + sqrt(S)) / (2.0*A);
    isp1 = o + d*t0;
    isp2 = o + d*t1;
    return true;
}

float get_height_modulation(float z0, float height)
{
    float h = abs(height / z0);
    float val = 1.0 / ((exp(h)+exp(-h))/2.0);
    float softEdge = 1.0 - smoothstep(HEIGHT_EDGE_FADE_START, HEIGHT_EDGE_FADE_END, h);
    return val*val*softEdge;
}

float get_winding(float rad, float winding_b, float winding_n)
{
    float r = rad + 0.05;
    return atan(exp(-0.25 / (0.5 * r)) / winding_b) * 2.0 * winding_n;
}

float find_difference(float t1, float t2)
{
    float v1 = abs(t1-t2);
    float v2 = abs(t1-t2-2.0*PI);
    float v3 = abs(t1-t2+2.0*PI);
    float v4 = abs(t1-t2-4.0*PI);
    float v5 = abs(t1-t2+4.0*PI);
    return min(v1, min(v2, min(v3, min(v4, v5))));
}

float get_arm(float rad, vec3 P, float disp, float delta, float arm, float winding_b, float winding_n)
{
    float workW = get_winding(rad, winding_b, winding_n);
    float workT = -atan(P.x, P.z) - delta;
    float v = abs(find_difference(workW, workT + disp)) / PI;
    return pow(clamp(1.0-v, 0.0, 1.0), arm * 15.0);
}

float arm_value(float rad, vec3 P, float delta, float arm, float no_arms, vec4 arms, float winding_b, float winding_n)
{
    int nArms = int(no_arms + 0.5);
    float v1 = get_arm(rad, P, arms.x, delta, arm, winding_b, winding_n);
    if (nArms == 1) return v1;
    float v = max(v1, get_arm(rad, P, arms.y, delta, arm, winding_b, winding_n));
    if (nArms == 2) return v;
    v = max(v, get_arm(rad, P, arms.z, delta, arm, winding_b, winding_n));
    if (nArms == 3) return v;
    return max(v, get_arm(rad, P, arms.w, delta, arm, winding_b, winding_n));
}

float sample_pos_dither(float tLower, float tUpper, float rnd)
{
    // Exact interpolation between the two actual neighboring samples on the ray.
    return mix(tLower, tUpper, clamp(rnd, 0.0, 1.0));
}

// Non-uniform step: s_n = s0*(1 + n*alpha), t_n = s0*n*(1 + alpha*(n-1)/2)
float stepIndexFromT(float t, float s0, float alpha)
{
    return alpha > 1e-6 ? (-1.0 + sqrt(1.0 + 2.0 * alpha * t / s0)) / alpha : t / s0;
}

float stepSizeAtN(float n, float s0, float alpha)
{
    return s0 * (1.0 + alpha * n);
}

float tAtN(float n, float s0, float alpha)
{
    return s0 * n * (1.0 + alpha * (n - 1.0) * 0.5);
}

SampleContribution sample_delta(vec3 p, float localStep, MarchParams mp)
{
    SampleContribution c;
    c.emit = vec3(0.0);
    c.trans = vec3(1.0);

    // Component id legend (cid):
    // 0: analytic disk / bulge emitter
    // 1: emissive fractal cloud (octave noise)
    // 2: absorbing cloud (octave noise attenuation)
    // 3: absorbing ridged cloud (ridged MF attenuation)
    // 4: emissive ridged cloud
    // 5: star field / clumpy emissive noise
    // Any other id: ignored

    for (int ci=0; ci<MAX_COMPONENTS; ++ci) {
        if (ci >= component_count()) break;

        ComponentParams cp;
        load_component(ci, cp);

        if (cp.isActive != 1.0) continue;

        vec3 P = vec3(p.x, 0.0, p.z);
        float radius = length(P) / mp.g.axis.x;
        float edgeFade = 1.0 - smoothstep(DISK_EDGE_FADE_START, DISK_EDGE_FADE_END, radius);
        vec3 pn = p / mp.g.axis;
        float volR = length(pn);
        float volumeFade = 1.0 - smoothstep(VOLUME_EDGE_FADE_START, VOLUME_EDGE_FADE_END, volR);
        edgeFade *= volumeFade;
        float z = get_height_modulation(cp.z0, p.y);

        if (cp.cid == 0) {
            float rho_0 = cp.strength * (localStep * MARCH_DENSITY_STEP_SCALE);
            float rad = (length(p) + 0.01) * cp.r0 + 0.01;
            float bi = rho_0 * (pow(rad, -0.855) * exp(-pow(rad, 0.25)) - 0.05);
            c.emit += max(bi, 0.0) * edgeFade * cp.spec * mp.rayStep;
            continue;
        }

        if (z <= 0.01) continue;
        float intensity = clamp(exp(-radius / (cp.r0 * 0.5)) - 0.01, 0.0, 1.0);
        intensity = min(intensity, 0.1);
        if (intensity <= 0.001) continue;

        float scale_inner = pow(smoothstep(0.0, 1.0*cp.inner, radius), 4.0);
        float armVal = 1.0;
        float local_winding = 0.0;
        if (cp.arm != 0.0) {
            armVal = arm_value(radius, P, cp.delta, cp.arm, mp.g.no_arms, mp.g.arms, mp.g.winding_b, mp.g.winding_n);
            if (cp.winding_mul != 0.0) local_winding = get_winding(radius, mp.g.winding_b, mp.g.winding_n) * cp.winding_mul;
        }

        float val = cp.strength * scale_inner * armVal * z * intensity;
        val *= edgeFade;
        float weighted = val * (localStep * MARCH_DENSITY_STEP_SCALE);
        if (weighted <= MARCH_WEIGHT_THRESHOLD) continue;

        if (cp.cid == 1) {
            vec3 pd = vec3(cos(local_winding*PI)*p.x + sin(local_winding*PI)*p.z, p.y, -sin(local_winding*PI)*p.x + cos(local_winding*PI)*p.z);
            float n = abs(octave_noise_3d(10.0, cp.ks, cp.scale*0.1, pd));
            n = pow(max(n, 0.01), cp.noise_tilt) + cp.noise_offset;
            if (n >= 0.0) c.emit += weighted * n * cp.spec * mp.rayStep;
        } else if (cp.cid == 2) {
            vec3 r = vec3(cos(local_winding*PI)*p.x + sin(local_winding*PI)*p.z, p.y, -sin(local_winding*PI)*p.x + cos(local_winding*PI)*p.z);
            float n = octave_noise_3d(9.0, cp.ks, cp.scale*0.1, r);
            n = clamp(pow(5.0 * max(n-cp.noise_offset, 0.0), cp.noise_tilt), -10.0, 10.0);
            vec3 att = exp(-n * weighted * cp.spec * 0.01);
            c.emit *= att;
            c.trans *= att;
        } else if (cp.cid == 3) {
            vec3 r = vec3(cos(local_winding*PI)*p.x + sin(local_winding*PI)*p.z, p.y, -sin(local_winding*PI)*p.x + cos(local_winding*PI)*p.z);
            float n = max(get_ridged_mf(r*cp.scale, cp.ks, 9, 2.5, cp.noise_offset, cp.noise_tilt), 0.0);
            vec3 att = exp(-n * weighted * cp.spec * 0.01);
            c.emit *= att;
            c.trans *= att;
        } else if (cp.cid == 4) {
            vec3 r = vec3(cos(local_winding*PI)*p.x + sin(local_winding*PI)*p.z, p.y, -sin(local_winding*PI)*p.x + cos(local_winding*PI)*p.z);
            float n = max(get_ridged_mf(r*cp.scale, cp.ks, 9, 2.5, cp.noise_offset, cp.noise_tilt), 0.0);
            c.emit += weighted * n * cp.spec * mp.rayStep;
        } else if (cp.cid == 5) {
            float perlin = abs(octave_noise_3d(10.0, cp.ks, cp.scale, p));
            float addN = 0.0;
            if (cp.noise_offset != 0.0) {
                vec3 r1 = vec3(cos(local_winding*PI)*p.x + sin(local_winding*PI)*p.z, p.y, -sin(local_winding*PI)*p.x + cos(local_winding*PI)*p.z);
                vec3 r2 = vec3(cos(local_winding*0.5*PI)*p.x + sin(local_winding*0.5*PI)*p.z, p.y, -sin(local_winding*0.5*PI)*p.x + cos(local_winding*0.5*PI)*p.z);
                addN = cp.noise_offset * octave_noise_3d(4.0, -2.0, 0.2, r1) + 0.5*cp.noise_offset*octave_noise_3d(4.0, -2.0, 0.4, r2);
            }
            float v = abs(pow(perlin + 1.0 + addN, cp.noise_tilt));
            c.emit += weighted * v * cp.spec * mp.rayStep;
        }
    }

    return c;
}

vec3 march_galaxy(MarchParams mp)
{
    vec3 I = vec3(0.0);
    float baseRnd = pseudo_blue_noise(mp.fragCoord);
    float thickness = max(mp.tFar - mp.t0, 0.0);
    float s0 = mp.step;
    float alpha = MARCH_STEP_ALPHA;
    float maxN = stepIndexFromT(thickness, s0, alpha);
    for (int iter=0; iter<MARCH_MAX_STEPS; ++iter) {
        float n = float(iter);
        if (n > maxN + 1.0) break;
        float dUpper = tAtN(n, s0, alpha);
        float dLower = tAtN(n + 1.0, s0, alpha);
        float cellUpper = mp.tFar - dUpper;
        if (cellUpper <= mp.t0) break;
        float cellLower = max(mp.tFar - dLower, mp.t0);
        float localStep = cellUpper - cellLower;
        if (localStep <= 0.0) break;
        float rnd = fract(baseRnd + float(iter) * 0.7548776662466927);
        float t = sample_pos_dither(cellLower, cellUpper, mix(0.5, rnd, MARCH_SAMPLE_DITHER));
        vec3 p = mp.camera + mp.dir * t;
        SampleContribution dI = sample_delta(p, localStep, mp);
        I = I * dI.trans + dI.emit;
        I = max(I, 0.0);
    }
    return I;
}

vec3 render_galaxy(vec3 camera, vec3 dir, vec2 fragCoord)
{
    GalaxyParams g;
    get_galaxy_params(g.axis, g.winding_b, g.winding_n, g.no_arms, g.arms);

    float rayStep = MARCH_BASE_STEP;
    float step = rayStep;
    vec3 I = vec3(0.0);

    vec3 isp1, isp2;
    float tNear, tFar;
    bool intersects = intersect_sphere(camera, dir, g.axis, isp1, isp2, tNear, tFar);
    if (intersects && tFar > 0.0) {
        MarchParams mp;
        mp.camera = camera;
        mp.dir = dir;
        mp.fragCoord = fragCoord;
        mp.t0 = max(tNear, 0.0);
        mp.tFar = tFar;
        mp.step = step;
        mp.rayStep = rayStep;
        mp.g = g;
        I = march_galaxy(mp);
    }

    I *= 0.01 / rayStep;
    return tanh(POST_TANH_STRENGTH * I);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    float az;
    float el;
    if (iMouse.z > 0.0) {
        vec2 m = iMouse.xy;
        az = (m.x / max(iResolution.x, 1.0) - 0.5) * 2.0 * PI;
        el = clamp((m.y / max(iResolution.y, 1.0) - CAMERA_MOUSE_EL_BIAS) * PI, -CAMERA_EL_CLAMP, CAMERA_EL_CLAMP);
    } else {
        az = CAMERA_IDLE_AZ_SPEED * iTime;
        el = CAMERA_IDLE_EL;
    }
    float radius = CAMERA_RADIUS;

    vec3 camPos = vec3(radius * cos(el) * sin(az), radius * sin(el), radius * cos(el) * cos(az));
    vec3 target = vec3(0.0);

    vec3 f = normalize(target - camPos);
    vec3 r = normalize(cross(vec3(0.0,1.0,0.0), f));
    vec3 u = cross(f, r);

    vec2 ndc = (2.0 * fragCoord - iResolution.xy) / max(iResolution.y, 1.0);
    float tanHalf = tan(radians(CAMERA_FOV_DEG) * 0.5);
    vec3 rayDir = normalize(f + ndc.x * r * tanHalf + ndc.y * u * tanHalf);

    vec3 col = render_galaxy(camPos, rayDir, fragCoord);
    fragColor = vec4(col, 1.0);
}



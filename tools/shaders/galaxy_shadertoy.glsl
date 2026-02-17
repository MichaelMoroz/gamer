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
#define GALAXY_PRESET PRESET_REDBAR // edit to choose preset

const float PI = 3.14159265;
const int MAX_COMPONENTS = 8;

const ivec3 grad3[16] = ivec3[16](
    ivec3(1,1,0), ivec3(-1,1,0), ivec3(1,-1,0), ivec3(-1,-1,0),
    ivec3(1,0,1), ivec3(-1,0,1), ivec3(1,0,-1), ivec3(-1,0,-1),
    ivec3(0,1,1), ivec3(0,-1,1), ivec3(0,1,-1), ivec3(0,-1,-1),
    ivec3(1,1,0), ivec3(-1,1,0), ivec3(1,-1,0), ivec3(-1,-1,0)
);

int fastfloor(float x) { return x > 0.0 ? int(x) : int(x) - 1; }
float dot3i(ivec3 g, float x, float y, float z) { return float(g.x)*x + float(g.y)*y + float(g.z)*z; }
float hash12(vec2 p) { return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715)))); }
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

void load_component(int i, out int cid, out vec4 p0, out vec4 p1, out vec4 p2, out vec3 spec)
{
#if GALAXY_PRESET == PRESET_SPIRAL
    if (i==0) { cid=0; p0=vec4(25.0,1.0,0.02,4.0); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.1); spec=spectrum_from_id(1); return; }
    if (i==1) { cid=1; p0=vec4(5000.0,0.3,0.02,0.4); p1=vec4(1.0,0.0,0.1,10.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==2) { cid=3; p0=vec4(150.0,0.2,0.03,0.45); p1=vec4(1.0,0.0,0.07,5.0); p2=vec4(1.0,1.5,1.0,0.2); spec=spectrum_from_id(2); return; }
    if (i==3) { cid=0; p0=vec4(30.0,1.0,0.02,10.0); p1=vec4(0.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(4); return; }
    if (i==4) { cid=5; p0=vec4(10.0,0.1,0.05,0.3); p1=vec4(1.0,0.0,0.0,6.0); p2=vec4(0.0,15.0,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==5) { cid=1; p0=vec4(1000.0,0.02,0.02,0.3); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(0); return; }
    cid=0; p0=vec4(0.0); p1=vec4(0.0); p2=vec4(0.0); spec=vec3(0.0); return;
#elif GALAXY_PRESET == PRESET_SOMBRERO
    if (i==0) { cid=0; p0=vec4(30.0,1.0,0.02,8.0); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.1); spec=spectrum_from_id(1); return; }
    if (i==1) { cid=1; p0=vec4(400.0,0.01,0.04,0.5); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,0.01,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==2) { cid=3; p0=vec4(50.0,0.1,0.02,0.4); p1=vec4(1.0,0.0,0.1,4.0); p2=vec4(1.2,1.5,1.0,0.6); spec=spectrum_from_id(2); return; }
    if (i==3) { cid=3; p0=vec4(30.0,0.5,0.02,0.15); p1=vec4(1.0,0.0,0.1,5.0); p2=vec4(1.1,1.5,1.0,0.05); spec=spectrum_from_id(2); return; }
    if (i==4) { cid=5; p0=vec4(3.0,0.1,0.05,0.5); p1=vec4(1.0,0.0,0.1,150.0); p2=vec4(0.0,12.0,1.0,0.0); spec=spectrum_from_id(0); return; }
    if (i==5) { cid=0; p0=vec4(5.0,0.1,0.05,4.0); p1=vec4(1.0,0.0,0.1,150.0); p2=vec4(0.0,12.0,1.0,0.0); spec=spectrum_from_id(0); return; }
    cid=0; p0=vec4(0.0); p1=vec4(0.0); p2=vec4(0.0); spec=vec3(0.0); return;
#elif GALAXY_PRESET == PRESET_SB0
    if (i==0) { cid=0; p0=vec4(250.0,1.0,0.02,25.0); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.1); spec=spectrum_from_id(1); return; }
    if (i==1) { cid=1; p0=vec4(5000.0,0.4,0.02,0.3); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==2) { cid=3; p0=vec4(500.0,0.9,0.015,0.35); p1=vec4(1.0,0.0,0.1,20.0); p2=vec4(1.0,1.2,1.0,0.02); spec=spectrum_from_id(2); return; }
    if (i==3) { cid=0; p0=vec4(30.0,1.0,0.02,10.0); p1=vec4(0.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(4); return; }
    if (i==4) { cid=5; p0=vec4(15.0,0.5,0.03,0.3); p1=vec4(1.0,0.0,0.0,150.0); p2=vec4(0.0,15.0,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==5) { cid=1; p0=vec4(1000.0,0.02,0.02,0.3); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(0); return; }
    cid=0; p0=vec4(0.0); p1=vec4(0.0); p2=vec4(0.0); spec=vec3(0.0); return;
#elif GALAXY_PRESET == PRESET_IRREGULAR
    if (i==0) { cid=1; p0=vec4(1500.0,0.1,0.02,0.5); p1=vec4(1.0,1.5,0.1,1.0); p2=vec4(0.0,0.2,1.0,0.2); spec=spectrum_from_id(2); return; }
    if (i==1) { cid=1; p0=vec4(500.0,0.1,0.05,0.4); p1=vec4(1.0,1.5,0.1,1.0); p2=vec4(0.0,0.3,1.0,0.0); spec=spectrum_from_id(0); return; }
    if (i==2) { cid=3; p0=vec4(10.0,0.05,0.025,0.45); p1=vec4(1.0,0.0,0.05,2.0); p2=vec4(1.5,1.5,1.0,0.3); spec=spectrum_from_id(2); return; }
    if (i==3) { cid=5; p0=vec4(5.0,0.05,0.02,0.4); p1=vec4(1.0,0.5,0.1,5.0); p2=vec4(0.0,15.0,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==4) { cid=0; p0=vec4(60.0,1.0,0.02,10.0); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(1); return; }
    if (i==5) { cid=5; p0=vec4(5.0,0.01,0.07,0.6); p1=vec4(1.0,0.0,0.1,100.0); p2=vec4(0.0,8.0,1.0,0.1); spec=spectrum_from_id(0); return; }
    if (i==6) { cid=3; p0=vec4(50.0,0.15,0.02,0.3); p1=vec4(1.0,2.5,0.35,7.0); p2=vec4(1.0,1.5,1.0,0.01); spec=spectrum_from_id(2); return; }
    cid=0; p0=vec4(0.0); p1=vec4(0.0); p2=vec4(0.0); spec=vec3(0.0); return;
#elif GALAXY_PRESET == PRESET_IRREGULAR2
    if (i==0) { cid=1; p0=vec4(1500.0,0.1,0.02,0.5); p1=vec4(1.0,1.5,0.1,1.0); p2=vec4(0.0,0.2,1.0,0.2); spec=spectrum_from_id(2); return; }
    if (i==1) { cid=1; p0=vec4(500.0,0.4,0.05,0.4); p1=vec4(1.0,1.5,0.1,1.0); p2=vec4(0.0,0.3,1.0,0.25); spec=spectrum_from_id(0); return; }
    if (i==2) { cid=3; p0=vec4(20.0,0.2,0.025,0.45); p1=vec4(1.0,0.0,0.5,5.0); p2=vec4(1.0,1.5,1.0,0.2); spec=spectrum_from_id(2); return; }
    if (i==3) { cid=5; p0=vec4(5.0,0.1,0.02,0.4); p1=vec4(1.0,0.5,0.1,5.0); p2=vec4(0.0,15.0,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==4) { cid=0; p0=vec4(250.0,0.1,0.02,15.0); p1=vec4(1.0,1.5,0.1,1.0); p2=vec4(0.0,0.2,1.0,0.2); spec=spectrum_from_id(3); return; }
    if (i==5) { cid=5; p0=vec4(5.0,0.01,0.07,0.6); p1=vec4(1.0,0.0,0.1,100.0); p2=vec4(0.0,8.0,1.0,0.1); spec=spectrum_from_id(0); return; }
    if (i==6) { cid=3; p0=vec4(25.0,0.35,0.02,0.3); p1=vec4(1.0,2.5,0.35,5.0); p2=vec4(1.1,1.5,1.0,0.01); spec=spectrum_from_id(2); return; }
    if (i==7) { cid=5; p0=vec4(15.0,0.1,0.04,0.5); p1=vec4(1.0,1.0,0.2,10.0); p2=vec4(0.0,10.0,1.0,0.2); spec=spectrum_from_id(1); return; }
    cid=0; p0=vec4(0.0); p1=vec4(0.0); p2=vec4(0.0); spec=vec3(0.0); return;
#elif GALAXY_PRESET == PRESET_REDBAR
    if (i==0) { cid=0; p0=vec4(13.0,1.0,0.03,3.0); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.1); spec=spectrum_from_id(1); return; }
    if (i==1) { cid=1; p0=vec4(500.0,0.1,0.03,0.45); p1=vec4(1.0,0.0,0.1,5.0); p2=vec4(0.0,0.5,-0.5,0.0); spec=spectrum_from_id(0); return; }
    if (i==2) { cid=0; p0=vec4(250.0,1.0,0.02,15.0); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==3) { cid=2; p0=vec4(150.0,0.2,0.01,0.45); p1=vec4(1.0,0.0,0.2,1.0); p2=vec4(0.0,1.0,-0.8,0.0); spec=spectrum_from_id(2); return; }
    if (i==4) { cid=5; p0=vec4(50.0,0.3,0.04,0.4); p1=vec4(1.0,0.0,0.1,30.0); p2=vec4(0.0,10.0,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==5) { cid=3; p0=vec4(300.0,2.0,0.02,5.0); p1=vec4(1.0,0.0,0.1,8.0); p2=vec4(1.0,1.5,1.0,0.0); spec=spectrum_from_id(2); return; }
    cid=0; p0=vec4(0.0); p1=vec4(0.0); p2=vec4(0.0); spec=vec3(0.0); return;
#elif GALAXY_PRESET == PRESET_TONSOFARMS
    if (i==0) { cid=0; p0=vec4(100.0,1.0,0.03,20.1); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.1); spec=spectrum_from_id(3); return; }
    if (i==1) { cid=1; p0=vec4(500.0,0.05,0.03,0.45); p1=vec4(1.0,0.0,0.1,5.0); p2=vec4(0.0,0.5,1.0,0.0); spec=spectrum_from_id(0); return; }
    if (i==2) { cid=5; p0=vec4(5.0,0.3,0.04,0.4); p1=vec4(1.0,0.0,0.1,50.0); p2=vec4(0.0,15.0,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==3) { cid=0; p0=vec4(20.0,1.0,0.02,5.0); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(1); return; }
    if (i==4) { cid=1; p0=vec4(3900.0,0.3,0.02,0.4); p1=vec4(1.0,0.0,0.1,3.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(2); return; }
    if (i==5) { cid=3; p0=vec4(150.0,0.5,0.015,0.4); p1=vec4(1.0,0.0,0.1,6.0); p2=vec4(1.05,1.5,1.0,0.25); spec=spectrum_from_id(2); return; }
    cid=0; p0=vec4(0.0); p1=vec4(0.0); p2=vec4(0.0); spec=vec3(0.0); return;
#elif GALAXY_PRESET == PRESET_VORTEXCLOUD
    if (i==0) { cid=0; p0=vec4(30.0,1.0,0.02,3.0); p1=vec4(1.0,0.0,0.5,1.0); p2=vec4(0.0,1.0,1.0,0.1); spec=spectrum_from_id(4); return; }
    if (i==1) { cid=3; p0=vec4(15.0,0.1,0.05,0.5); p1=vec4(1.0,0.0,0.2,2.0); p2=vec4(1.3,1.0,1.0,0.25); spec=spectrum_from_id(2); return; }
    if (i==2) { cid=1; p0=vec4(100.0,0.01,0.03,0.6); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,0.01,1.0,0.0); spec=spectrum_from_id(0); return; }
    cid=0; p0=vec4(0.0); p1=vec4(0.0); p2=vec4(0.0); spec=vec3(0.0); return;
#elif GALAXY_PRESET == PRESET_WHEELGALAXY
    if (i==0) { cid=0; p0=vec4(30.0,1.0,0.02,5.0); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,1.0,1.0,0.0); spec=spectrum_from_id(1); return; }
    if (i==1) { cid=1; p0=vec4(900.0,0.01,0.03,0.4); p1=vec4(1.0,0.0,0.1,1.0); p2=vec4(0.0,0.4,1.0,0.5); spec=spectrum_from_id(2); return; }
    if (i==2) { cid=3; p0=vec4(250.0,0.25,0.02,0.4); p1=vec4(1.0,0.0,0.1,10.0); p2=vec4(1.0,1.1,1.0,0.6); spec=spectrum_from_id(2); return; }
    if (i==3) { cid=5; p0=vec4(0.5,0.05,0.02,0.5); p1=vec4(1.0,0.0,0.1,6.0); p2=vec4(0.0,20.0,1.0,0.3); spec=spectrum_from_id(2); return; }
    if (i==4) { cid=5; p0=vec4(10.0,0.1,0.02,0.55); p1=vec4(1.0,0.0,0.1,100.0); p2=vec4(0.0,10.0,1.0,0.3); spec=spectrum_from_id(0); return; }
    if (i==5) { cid=3; p0=vec4(50.0,0.2,0.02,0.2); p1=vec4(1.0,0.0,0.25,5.0); p2=vec4(1.1,1.0,1.0,0.05); spec=spectrum_from_id(2); return; }
    cid=0; p0=vec4(0.0); p1=vec4(0.0); p2=vec4(0.0); spec=vec3(0.0); return;
#else
    cid=0; p0=vec4(0.0); p1=vec4(0.0); p2=vec4(0.0); spec=vec3(0.0); return;
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
    if (h > 2.0) return 0.0;
    float val = 1.0 / ((exp(h)+exp(-h))/2.0);
    return val*val;
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

vec3 post_process(vec3 v, float exposure, float gamma, float saturation)
{
    v *= 1.0 / max(exposure, 1e-4);
    v = pow(max(v, 0.0), vec3(gamma));
    float center = (v.x+v.y+v.z)/3.0;
    vec3 tmp = vec3(center) - v;
    v = vec3(center) - saturation * tmp;
    v = clamp(v * 10.0, 0.0, 255.0);
    return v / 255.0;
}

vec3 render_galaxy(vec3 camera, vec3 dir, vec2 fragCoord)
{
    vec3 axis;
    float winding_b, winding_n, no_arms;
    vec4 arms;
    get_galaxy_params(axis, winding_b, winding_n, no_arms, arms);

    vec3 I = vec3(0.0);
    float rayStep = 0.025;

    vec3 isp1, isp2;
    float tNear, tFar;
    bool intersects = intersect_sphere(camera, dir, axis, isp1, isp2, tNear, tFar);
    if (intersects && tFar > 0.0) {
        float t0 = max(tNear, 0.0);
        float t = tFar;
        float step = rayStep;
        float dither = hash12(fragCoord);
        t -= dither * step;
        t = clamp(t, t0, tFar);
        vec3 p = camera + dir * t;

        for (int iter=0; iter<512; ++iter) {
            if (t <= t0 - step) break;
            step = clamp(length(p-camera) * rayStep, 0.001, 0.01);

            for (int ci=0; ci<MAX_COMPONENTS; ++ci) {
                if (ci >= component_count()) break;

                int cid;
                vec4 p0, p1, p2;
                vec3 spec;
                load_component(ci, cid, p0, p1, p2, spec);

                float strength = p0.x;
                float arm = p0.y;
                float z0 = p0.z;
                float r0 = p0.w;

                float isActive = p1.x;
                float delta = p1.y;
                float winding_mul = p1.z;
                float scale = p1.w;

                float noise_offset = p2.x;
                float noise_tilt = p2.y;
                float ks = p2.z;
                float inner = p2.w;

                if (isActive != 1.0) continue;

                vec3 P = vec3(p.x, 0.0, p.z);
                float radius = length(P) / axis.x;
                float z = get_height_modulation(z0, p.y);

                if (cid == 0) {
                    float rho_0 = strength * (step * 200.0);
                    float rad = (length(p) + 0.01) * r0 + 0.01;
                    float bi = rho_0 * (pow(rad, -0.855) * exp(-pow(rad, 0.25)) - 0.05);
                    I += max(bi, 0.0) * spec * rayStep;
                    continue;
                }

                if (z <= 0.01) continue;
                float intensity = clamp(exp(-radius / (r0 * 0.5)) - 0.01, 0.0, 1.0);
                intensity = min(intensity, 0.1);
                if (intensity <= 0.001) continue;

                float scale_inner = pow(smoothstep(0.0, 1.0*inner, radius), 4.0);
                float armVal = 1.0;
                float local_winding = 0.0;
                if (arm != 0.0) {
                    armVal = arm_value(radius, P, delta, arm, no_arms, arms, winding_b, winding_n);
                    if (winding_mul != 0.0) local_winding = get_winding(radius, winding_b, winding_n) * winding_mul;
                }

                float val = strength * scale_inner * armVal * z * intensity;
                float weighted = val * (step * 200.0);
                if (weighted <= 0.0005) continue;

                if (cid == 1) {
                    vec3 pd = vec3(cos(local_winding*PI)*p.x + sin(local_winding*PI)*p.z, p.y, -sin(local_winding*PI)*p.x + cos(local_winding*PI)*p.z);
                    float n = abs(octave_noise_3d(10.0, ks, scale*0.1, pd));
                    n = pow(max(n, 0.01), noise_tilt) + noise_offset;
                    if (n >= 0.0) I += weighted * n * spec * rayStep;
                } else if (cid == 2) {
                    vec3 r = vec3(cos(local_winding*PI)*p.x + sin(local_winding*PI)*p.z, p.y, -sin(local_winding*PI)*p.x + cos(local_winding*PI)*p.z);
                    float n = octave_noise_3d(9.0, ks, scale*0.1, r);
                    n = clamp(pow(5.0 * max(n-noise_offset, 0.0), noise_tilt), -10.0, 10.0);
                    I *= exp(-n * weighted * spec * 0.01);
                } else if (cid == 3) {
                    vec3 r = vec3(cos(local_winding*PI)*p.x + sin(local_winding*PI)*p.z, p.y, -sin(local_winding*PI)*p.x + cos(local_winding*PI)*p.z);
                    float n = max(get_ridged_mf(r*scale, ks, 9, 2.5, noise_offset, noise_tilt), 0.0);
                    I *= exp(-n * weighted * spec * 0.01);
                } else if (cid == 4) {
                    vec3 r = vec3(cos(local_winding*PI)*p.x + sin(local_winding*PI)*p.z, p.y, -sin(local_winding*PI)*p.x + cos(local_winding*PI)*p.z);
                    float n = max(get_ridged_mf(r*scale, ks, 9, 2.5, noise_offset, noise_tilt), 0.0);
                    I += weighted * n * spec * rayStep;
                } else if (cid == 5) {
                    float perlin = abs(octave_noise_3d(10.0, ks, scale, p));
                    float addN = 0.0;
                    if (noise_offset != 0.0) {
                        vec3 r1 = vec3(cos(local_winding*PI)*p.x + sin(local_winding*PI)*p.z, p.y, -sin(local_winding*PI)*p.x + cos(local_winding*PI)*p.z);
                        vec3 r2 = vec3(cos(local_winding*0.5*PI)*p.x + sin(local_winding*0.5*PI)*p.z, p.y, -sin(local_winding*0.5*PI)*p.x + cos(local_winding*0.5*PI)*p.z);
                        addN = noise_offset * octave_noise_3d(4.0, -2.0, 0.2, r1) + 0.5*noise_offset*octave_noise_3d(4.0, -2.0, 0.4, r2);
                    }
                    float v = abs(pow(perlin + 1.0 + addN, noise_tilt));
                    I += weighted * v * spec * rayStep;
                }
            }

            p -= dir * step;
            t -= step;
            I = max(I, 0.0);
        }
    }

    I *= 0.01 / rayStep;
    return post_process(I, 1.0, 1.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    float az;
    float el;
    if (iMouse.z > 0.0) {
        vec2 m = iMouse.xy;
        az = (m.x / max(iResolution.x, 1.0) - 0.5) * 2.0 * PI;
        el = clamp((m.y / max(iResolution.y, 1.0) - 0.3) * PI, -1.45, 1.45);
    } else {
        az = 0.35 * iTime;
        el = 0.2;
    }
    float radius = 1.2;

    vec3 camPos = vec3(radius * cos(el) * sin(az), radius * sin(el), radius * cos(el) * cos(az));
    vec3 target = vec3(0.0);

    vec3 f = normalize(target - camPos);
    vec3 r = normalize(cross(vec3(0.0,1.0,0.0), f));
    vec3 u = cross(f, r);

    vec2 ndc = (2.0 * fragCoord - iResolution.xy) / max(iResolution.y, 1.0);
    float tanHalf = tan(radians(60.0) * 0.5);
    vec3 rayDir = normalize(f + ndc.x * r * tanHalf + ndc.y * u * tanHalf);

    vec3 col = render_galaxy(camPos, rayDir, fragCoord);
    fragColor = vec4(col, 1.0);
}

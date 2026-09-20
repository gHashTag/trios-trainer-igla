// Replica of trios_trainer::det_math::exp_det with parameterizable constants.
// It is FIRST verified to reproduce the frozen vector exactly with the pristine
// constants; a replica that does not is not evidence about the real function.
#[derive(Clone, Copy)]
struct K { log2e: f32, ln2hi: f32, ln2lo: f32, c2: f32, c3: f32, c4: f32, c5: f32, c6: f32 }
const P: K = K { log2e: 1.442_695_04, ln2hi: 0.693_145_75, ln2lo: 1.428_606_8e-6,
                 c2: 0.5, c3: 0.166_666_67, c4: 0.041_666_67, c5: 0.008_333_33, c6: 0.001_388_89 };
fn exp_det(x: f32, k: K) -> f32 {
    if x.is_nan() { return x; }
    if x > 88.722_84 { return f32::INFINITY; }
    let half = if x >= 0.0 { 0.5 } else { -0.5 };
    let e = (x * k.log2e + half) as i32;
    if e < -126 { return 0.0; }
    let kf = e as f32;
    let r = (x - kf * k.ln2hi) - kf * k.ln2lo;
    let p = 1.0 + r * (1.0 + r * (k.c2 + r * (k.c3 + r * (k.c4 + r * (k.c5 + r * k.c6)))));
    let (k1, k2) = if e > 127 { (127, e - 127) } else { (e, 0) };
    let s1 = f32::from_bits(((k1 + 127) as u32) << 23);
    let s2 = f32::from_bits(((k2 + 127) as u32) << 23);
    (p * s1) * s2
}
const FROZEN: &[(u32, u32)] = &[
 (0x00000000,0x3f800000),(0x80000000,0x3f800000),(0x3f800000,0x402df854),(0xbf800000,0x3ebc5ab2),
 (0x3f000000,0x3fd3094c),(0xbf000000,0x3f1b4598),(0x40000000,0x40ec7326),(0xc0000000,0x3e0a9555),
 (0x40e00000,0x44891443),(0xc0e00000,0x3a6f0b5d),(0x3f317218,0x40000000),(0xbf317218,0x3f000000),
 (0x322bcc77,0x3f800000),(0xb22bcc77,0x3f800000),(0x3dcccccd,0x3f8d763e),(0xbdcccccd,0x3f67a36d),
 (0x40490fdb,0x41b92026),(0xc0490fdb,0x3d310112),(0x41380000,0x47c0cde3),(0xc1380000,0x3729f46c),
 (0x42ae0000,0x7e36d80a),(0xc2ae0000,0x00b33686),(0x3eb17218,0x3fb504f5),(0xbeb17218,0x3f3504f2),
 (0x41a20000,0x4e1486bc),(0xc1a20000,0x30dc9ef1),(0x0da24260,0x3f800000),(0x8da24260,0x3f800000)];
fn up(v: f32) -> f32 { f32::from_bits(v.to_bits() + 1) }
fn frozen_flips(k: K) -> usize {
    FROZEN.iter().filter(|(i,o)| exp_det(f32::from_bits(*i), k).to_bits() != *o).count()
}
fn sweep_flips(k: K) -> (usize, usize, f32) {
    let n = 400_000u32; let mut flips = 0; let mut first = f32::NAN;
    for i in 0..=n {
        let x = -8.0 + 16.0 * (i as f32) / (n as f32);
        if exp_det(x, k).to_bits() != exp_det(x, P).to_bits() {
            if flips == 0 { first = x; }
            flips += 1;
        }
    }
    (n as usize + 1, flips, first)
}
fn main() {
    // 1. the replica must BE the function.
    assert_eq!(frozen_flips(P), 0, "replica does not reproduce the frozen vector");
    println!("replica reproduces all {} frozen pairs with pristine constants", FROZEN.len());
    println!();
    println!("{:<8} {:>12} {:>10} {:>14}  {}", "const", "frozen/28", "sweep", "of 400001", "first differing x");
    let cases: &[(&str, fn(K) -> K)] = &[
        ("LOG2E", |mut k| { k.log2e = up(k.log2e); k }),
        ("LN2_HI", |mut k| { k.ln2hi = up(k.ln2hi); k }),
        ("LN2_LO", |mut k| { k.ln2lo = up(k.ln2lo); k }),
        ("C2", |mut k| { k.c2 = up(k.c2); k }),
        ("C3", |mut k| { k.c3 = up(k.c3); k }),
        ("C4", |mut k| { k.c4 = up(k.c4); k }),
        ("C5", |mut k| { k.c5 = up(k.c5); k }),
        ("C6", |mut k| { k.c6 = up(k.c6); k }),
    ];
    for (name, f) in cases {
        let k = f(P);
        let fr = frozen_flips(k);
        let (n, sw, first) = sweep_flips(k);
        println!("{:<8} {:>12} {:>10} {:>14}  {}", name, fr, sw, n, if sw > 0 { format!("{}", first) } else { "-".into() });
    }
}

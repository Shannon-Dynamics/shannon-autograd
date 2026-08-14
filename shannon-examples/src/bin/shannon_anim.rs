//! W2 — real-time animated demo: a robot arm fixing the "S H A N N O N" sign.
//!
//! Loop: the H lies dropped on the table → the arm picks it up → places it in
//! its slot → waves to the camera → bumps the H on the way back → it falls →
//! repeat. Camera fixed, front-on.
//!
//! ALL animation runs on the host (state machine + two-link IK); the kernel is
//! a pure function of per-frame transforms.
//!
//! Modes:
//!   (default)              live minifb window
//!   --frames N             headless: render N frames, print fps stats
//!   --still <t> <name>     headless: render one frame at loop-time t → name.ppm/.png
//!   --width W --height H   resolution (default 960×540)

use anyhow::Result;
use shannon_core::scene_shannon::{ARM_L1, ARM_L2, ARM_SHOULDER, CAM_TILT, H_SLOT, LETTER_HH, LETTER_Y, LETTER_Z, SLOT_X, TABLE_TOP_Y};
use shannon_core::{Quat, Vec3};
use shannon_examples::image;
use shannon_rt::{Array, launch};
use std::time::Instant;

// ── Host-side animation helpers ─────────────────────────────────────────────

fn smooth(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

fn lerp3(a: Vec3, b: Vec3, u: f32) -> Vec3 {
    a + (b - a) * u
}

/// Normalized quaternion lerp — fine for the small-to-moderate blends here.
fn nlerp(a: Quat, b: Quat, u: f32) -> Quat {
    let dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    let s = if dot < 0.0 { -1.0 } else { 1.0 };
    Quat::new(
        a.x + (b.x * s - a.x) * u,
        a.y + (b.y * s - a.y) * u,
        a.z + (b.z * s - a.z) * u,
        a.w + (b.w * s - a.w) * u,
    )
    .normalize()
}

/// Two-link IK: wrist chases `target` from the fixed shoulder.
/// Returns (elbow, wrist). Elbow-up solution via the axis ⟂ (reach, up).
fn ik(target: Vec3) -> (Vec3, Vec3) {
    let s0 = ARM_SHOULDER;
    let mut dir = target - s0;
    let mut dist = dir.length();
    let max_reach = 0.98 * (ARM_L1 + ARM_L2);
    if dist < 0.15 {
        dist = 0.15;
    }
    if dist > max_reach {
        dist = max_reach;
    }
    dir = dir.normalize();
    let t_eff = s0 + dir * dist;

    let cos_a = ((ARM_L1 * ARM_L1 + dist * dist - ARM_L2 * ARM_L2) / (2.0 * ARM_L1 * dist)).clamp(-1.0, 1.0);
    let a = cos_a.acos();

    let up = Vec3::new(0.0, 1.0, 0.0);
    let raw_axis = dir.cross(up);
    let axis = if raw_axis.length_sq() > 1.0e-6 { raw_axis.normalize() } else { Vec3::new(1.0, 0.0, 0.0) };
    let elbow_dir = Quat::from_axis_angle(axis, -a).rotate(dir); // −a ⇒ elbow up
    let s1 = s0 + elbow_dir * ARM_L1;
    (s1, t_eff)
}

// ── The animation script ────────────────────────────────────────────────────

const LOOP_T: f32 = 16.0;

/// Where the H rests when dropped: flat on the tabletop, in front of the row.
fn drop_pose() -> (Vec3, Quat) {
    let pos = Vec3::new(-0.50, TABLE_TOP_Y + 0.035, 0.30);
    let rot = Quat::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), 0.35)
        .mul(Quat::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), -core::f32::consts::FRAC_PI_2));
    (pos, rot)
}

fn slot_pose() -> (Vec3, Quat) {
    (Vec3::new(SLOT_X[H_SLOT], LETTER_Y, LETTER_Z), Quat::IDENTITY)
}

struct Frame {
    s1: Vec3,
    s2: Vec3,
    grip_dir: Vec3,
    grip: f32,
    h_pos: Vec3,
    h_rot: Quat,
}

/// Wrist target that holds the letter by its top bar.
fn wrist_for_held(h_pos: Vec3, h_rot: Quat) -> Vec3 {
    h_pos + h_rot.rotate(Vec3::new(0.0, LETTER_HH + 0.02, 0.0)) + Vec3::new(0.0, 0.13, 0.0)
}

fn pose(t: f32) -> Frame {
    let (h_drop, r_drop) = drop_pose();
    let (h_slot, r_slot) = slot_pose();
    let home = Vec3::new(0.9, 1.35, 0.1);
    let hover = h_drop + Vec3::new(0.0, 0.42, 0.0);
    let grab = h_drop + Vec3::new(0.0, 0.17, 0.0);
    let wave_base = Vec3::new(0.3, 1.95, 0.45);

    // Defaults: H dropped, arm home, open grip.
    let mut h_pos = h_drop;
    let mut h_rot = r_drop;
    let mut grip = 0.0;
    let wrist;

    if t < 2.2 {
        // REACH: home → hover above the dropped H.
        wrist = lerp3(home, hover, smooth(t / 2.2));
    } else if t < 3.0 {
        // DESCEND onto it.
        wrist = lerp3(hover, grab, smooth((t - 2.2) / 0.8));
    } else if t < 3.6 {
        // GRAB.
        wrist = grab;
        grip = smooth((t - 3.0) / 0.6);
    } else if t < 4.6 {
        // LIFT — H starts following, begins to rotate upright.
        let u = smooth((t - 3.6) / 1.0);
        h_pos = lerp3(h_drop, h_drop + Vec3::new(0.0, 0.5, 0.0), u);
        h_rot = nlerp(r_drop, r_slot, 0.35 * u);
        grip = 1.0;
        wrist = wrist_for_held(h_pos, h_rot);
    } else if t < 6.4 {
        // TRAVERSE to above the slot, finishing upright.
        let u = smooth((t - 4.6) / 1.8);
        let lifted = h_drop + Vec3::new(0.0, 0.5, 0.0);
        let above_slot = h_slot + Vec3::new(0.0, 0.45, 0.0);
        h_pos = lerp3(lifted, above_slot, u);
        h_rot = nlerp(nlerp(r_drop, r_slot, 0.35), r_slot, u);
        grip = 1.0;
        wrist = wrist_for_held(h_pos, h_rot);
    } else if t < 7.2 {
        // LOWER into the slot.
        let u = smooth((t - 6.4) / 0.8);
        h_pos = lerp3(h_slot + Vec3::new(0.0, 0.45, 0.0), h_slot, u);
        h_rot = r_slot;
        grip = 1.0;
        wrist = wrist_for_held(h_pos, h_rot);
    } else if t < 7.8 {
        // RELEASE and lift away.
        let u = smooth((t - 7.2) / 0.6);
        h_pos = h_slot;
        h_rot = r_slot;
        grip = 1.0 - u;
        wrist = wrist_for_held(h_slot, r_slot) + Vec3::new(0.0, 0.3 * u, 0.0);
    } else if t < 8.6 {
        // Move up toward the wave position.
        let u = smooth((t - 7.8) / 0.8);
        h_pos = h_slot;
        h_rot = r_slot;
        wrist = lerp3(wrist_for_held(h_slot, r_slot) + Vec3::new(0.0, 0.3, 0.0), wave_base, u);
    } else if t < 11.4 {
        // WAVE to the camera.
        let u = t - 8.6;
        h_pos = h_slot;
        h_rot = r_slot;
        wrist = wave_base + Vec3::new(0.38 * (u * 2.0 * core::f32::consts::PI * 0.9).sin(), 0.0, 0.0);
        grip = 0.15;
    } else if t < 12.4 {
        // BUMP: the retract path deliberately clips the H's top corner.
        let u = smooth((t - 11.4) / 1.0);
        let bump_pt = Vec3::new(SLOT_X[H_SLOT] + 0.10, LETTER_Y + 0.16, LETTER_Z + 0.08);
        wrist = if u < 0.55 {
            lerp3(wave_base, bump_pt, u / 0.55)
        } else {
            lerp3(bump_pt, home, (u - 0.55) / 0.45)
        };
        // H stays in the slot until contact at u ≈ 0.55.
        if u < 0.55 {
            h_pos = h_slot;
            h_rot = r_slot;
        } else {
            let v = ((t - 11.4) - 0.55) / (LOOP_T - 11.95);
            let (hp, hr) = fall(h_slot, r_slot, h_drop, r_drop, v * (LOOP_T - 11.95) / 1.1);
            h_pos = hp;
            h_rot = hr;
        }
    } else {
        // FALL continues; arm returns home.
        let v = (t - 11.95) / 1.1;
        let (hp, hr) = fall(h_slot, r_slot, h_drop, r_drop, v);
        h_pos = hp;
        h_rot = hr;
        wrist = lerp3(Vec3::new(SLOT_X[H_SLOT] + 0.10, LETTER_Y + 0.16, LETTER_Z + 0.08), home, smooth((t - 12.4) / 1.2));
    }

    let (s1, s2) = ik(wrist);
    let grip_dir = (s2 - s1).normalize();
    Frame { s1, s2, grip_dir, grip, h_pos, h_rot }
}

/// The H topples off the front edge back to its drop pose. `u` in 0..1+.
fn fall(from_p: Vec3, from_r: Quat, to_p: Vec3, to_r: Quat, u: f32) -> (Vec3, Quat) {
    let u = u.clamp(0.0, 1.0);
    let ease = u * u; // accelerating — reads as gravity
    // Horizontal travel linear-ish, vertical with a small pop then drop.
    let mut p = lerp3(from_p, to_p, u);
    p.y = from_p.y + (to_p.y - from_p.y) * ease + 0.18 * (u * core::f32::consts::PI).sin() * (1.0 - u);
    (p, nlerp(from_r, to_r, smooth(ease)))
}

// ── Rendering plumbing ──────────────────────────────────────────────────────

fn render(frame: &Frame, cam_rot: Quat, w: u32, h: u32, pixels: &mut Array<Vec3>) -> Result<Vec<Vec3>> {
    let n = (w * h) as usize;
    launch!(
        draw_shannon,
        dim = n,
        (frame.s1, frame.s2, frame.grip_dir, frame.grip, frame.h_pos, frame.h_rot, cam_rot, w, h, &mut *pixels)
    )?;
    pixels.to_vec()
}

struct Args {
    width: u32,
    height: u32,
    frames: Option<u32>,
    still: Option<(f32, String)>,
}

fn parse_args() -> Args {
    let mut a = Args { width: 960, height: 540, frames: None, still: None };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--width" => {
                a.width = argv[i + 1].parse().expect("--width N");
                i += 2;
            }
            "--height" => {
                a.height = argv[i + 1].parse().expect("--height N");
                i += 2;
            }
            "--frames" => {
                a.frames = Some(argv[i + 1].parse().expect("--frames N"));
                i += 2;
            }
            "--still" => {
                a.still = Some((argv[i + 1].parse().expect("--still <t> <name>"), argv[i + 2].clone()));
                i += 3;
            }
            "--parity" => i += 1,
            other => panic!("unknown argument {other}"),
        }
    }
    a
}

fn main() -> Result<()> {
    let args = parse_args();
    let cam_rot = Quat::from_rpy(CAM_TILT, 0.0, 0.0);
    let n = (args.width * args.height) as usize;
    let mut pixels = Array::<Vec3>::zeros(n)?;

    // ---- still frame ----
    if let Some((t, name)) = &args.still {
        let frame = pose(t % LOOP_T);
        let buf = render(&frame, cam_rot, args.width, args.height, &mut pixels)?;
        image::write_ppm(&format!("{name}.ppm"), &buf, args.width, args.height)?;
        image::write_png(&format!("{name}.png"), &buf, args.width, args.height)?;
        println!("✓ wrote {name}.ppm / {name}.png (t = {t})");
        return Ok(());
    }

    // ---- CPU/GPU parity (A6) ----
    if std::env::args().any(|a| a == "--parity") {
        const PW: u32 = 128;
        const PH: u32 = 72;
        let pn = (PW * PH) as usize;
        let mut small = Array::<Vec3>::zeros(pn)?;
        let frame = pose(5.5); // mid-carry: arm, held H, letters all in frame
        launch!(
            draw_shannon,
            dim = pn,
            (frame.s1, frame.s2, frame.grip_dir, frame.grip, frame.h_pos, frame.h_rot, cam_rot, PW, PH, &mut small)
        )?;
        let a = small.to_vec()?;
        let mut b = vec![Vec3::ZERO; pn];
        shannon_cpu::draw_shannon(
            frame.s1, frame.s2, frame.grip_dir, frame.grip, frame.h_pos, frame.h_rot, cam_rot, PW, PH, &mut b,
        );
        let (mut worst, mut over) = (0.0f32, 0usize);
        for i in 0..pn {
            let d = a[i] - b[i];
            let m = d.x.abs().max(d.y.abs()).max(d.z.abs());
            worst = worst.max(m);
            if m > 1e-3 {
                over += 1;
            }
        }
        // Same robust criterion as W1 (Day-2 plan): branch-boundary pixels may differ.
        assert!(over as f32 / pn as f32 <= 0.005, "{over}/{pn} pixels exceed 1e-3");
        assert!(worst <= 5e-2, "worst channel delta {worst}");
        println!("✓ CPU == GPU at {PW}×{PH} ({:.2}% >1e-3, worst {worst:.2e})", 100.0 * over as f32 / pn as f32);
        return Ok(());
    }

    // ---- headless benchmark ----
    if let Some(count) = args.frames {
        // Warm-up: first launch pays the one-time PTX module load (~0.85 s).
        // Steady-state fps is the meaningful number for a render loop.
        let _ = render(&pose(0.0), cam_rot, args.width, args.height, &mut pixels)?;
        let start = Instant::now();
        for k in 0..count {
            let t = (k as f32 / 30.0) % LOOP_T; // simulate 30 fps timeline
            let frame = pose(t);
            let _ = render(&frame, cam_rot, args.width, args.height, &mut pixels)?;
        }
        let dt = start.elapsed().as_secs_f32();
        println!(
            "✓ {count} frames at {}×{} in {dt:.2}s → {:.1} fps",
            args.width,
            args.height,
            count as f32 / dt
        );
        return Ok(());
    }

    // ---- live window ----
    use minifb::{Key, Window, WindowOptions};
    let mut window = Window::new(
        "shannon-autograd — live SDF demo (Esc quits)",
        args.width as usize,
        args.height as usize,
        WindowOptions::default(),
    )?;
    window.set_target_fps(60);
    eprintln!("[w2] window opened — rendering. Esc to quit.");

    let mut u32buf = vec![0u32; n];
    let start = Instant::now();
    let mut frames = 0u32;
    let mut last_report = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t = start.elapsed().as_secs_f32() % LOOP_T;
        let frame = pose(t);
        let buf = render(&frame, cam_rot, args.width, args.height, &mut pixels)?;

        // Vec3 [0,1] → 0RGB, flipped vertically (buffer row 0 = bottom).
        for y in 0..args.height as usize {
            let src = (args.height as usize - 1 - y) * args.width as usize;
            for x in 0..args.width as usize {
                let p = buf[src + x];
                let r = (p.x.clamp(0.0, 1.0) * 255.0) as u32;
                let g = (p.y.clamp(0.0, 1.0) * 255.0) as u32;
                let b = (p.z.clamp(0.0, 1.0) * 255.0) as u32;
                u32buf[y * args.width as usize + x] = (r << 16) | (g << 8) | b;
            }
        }
        window.update_with_buffer(&u32buf, args.width as usize, args.height as usize)?;

        frames += 1;
        if last_report.elapsed().as_secs_f32() > 2.0 {
            println!("{:.1} fps", frames as f32 / last_report.elapsed().as_secs_f32());
            frames = 0;
            last_report = Instant::now();
        }
    }
    Ok(())
}

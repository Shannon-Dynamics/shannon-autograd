//! ARM-7 pick & place — an industrial 4-DOF robot arm (yaw / shoulder /
//! elbow / wrist + two-finger gripper) restores the fallen H of the flat
//! "S H A N N O N" sign, waves to the camera, then bumps the H off the table
//! on its way home so the loop repeats seamlessly. The arm model and staging
//! port `docs/robot-arm-shannon.html`; the objective (fix the sign → greet →
//! undo it) matches the original `shannon_anim` demo.
//!
//! ALL animation runs on the host: a keyframe timeline over the four joint
//! angles + grip, forward kinematics to joint positions, and the H's carry /
//! snap / tumble. The kernel is a pure function of per-frame positions.
//!
//! Modes:
//!   (default)              live minifb window
//!   --frames N             headless: render N frames, print fps stats
//!   --still <t> <name>     headless: render one frame at loop-time t → name.ppm/.png
//!   --parity               CPU == GPU spot check at 128×72
//!   --width W --height H   resolution (default 960×540)

use anyhow::Result;
use shannon_core::Vec3;
use shannon_core::scene_arm7::{
    ARM_L1, ARM_L2, ARM_L3, ARM_L4, H_SLOT_X, LETTER_Y, SHOULDER, TABLE_Z, finger_spread,
};
use shannon_examples::image;
use shannon_kernels::launch;
use shannon_rt::Array;
use std::time::Instant;

const LOOP_T: f32 = 16.0;

// Default orbit camera (the HTML's home view).
const CAM_AZ: f32 = 0.55;
const CAM_EL: f32 = 0.62;
const CAM_DIST: f32 = 4.3;

/// Where the H lies after tumbling off: flat on the floor in front of the table.
const DROP_POS: Vec3 = Vec3::new(-0.95, 0.03, 0.70);
const DROP_YAW: f32 = 42.0 * core::f32::consts::PI / 180.0;
/// The H's slot in the sign.
const SLOT_POS: Vec3 = Vec3::new(H_SLOT_X, LETTER_Y, TABLE_Z);

const T_GRAB: f32 = 3.6;
const T_YAW_END: f32 = 7.2;
const T_SNAP_START: f32 = 7.2;
const T_RELEASE: f32 = 7.9;
/// The retract sweep clips the H at this moment; it tumbles back to the floor.
const T_BUMP_HIT: f32 = 14.05;
const FALL_DUR: f32 = 1.1;

fn smooth(u: f32) -> f32 {
    let u = u.clamp(0.0, 1.0);
    u * u * (3.0 - 2.0 * u)
}

fn lerp3(a: Vec3, b: Vec3, u: f32) -> Vec3 {
    a + (b - a) * u
}

// ── Kinematics: 3-segment arm driven by a hand target ───────────────────────
// The base yaws toward the target's azimuth; the three links (L1, L2, L3)
// bend in the vertical plane through that azimuth — the classic elbow-up
// two-link solution with the L2+L3 pair treated as the forearm.

struct Joints {
    elbow: Vec3,
    wrist: Vec3,
    hand_end: Vec3,
    f1: Vec3,
    f2: Vec3,
    /// Direction the hand (and fingers) point — used to hang the carried H.
    tip: Vec3,
}

fn ik(target: Vec3, grip: f32) -> Joints {
    let up = Vec3::new(0.0, 1.0, 0.0);
    // Azimuth: the arm faces +z at yaw 0 (toward the table).
    let rel = target - SHOULDER;
    let mut r = Vec3::new(rel.x, 0.0, rel.z);
    r = if r.length_sq() > 1.0e-6 {
        r.normalize()
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };

    // Planar coordinates in the (r, up) plane.
    let a = ARM_L1;
    let b = ARM_L2 + ARM_L3;
    let mut dr = rel.x * r.x + rel.z * r.z;
    let mut dy = rel.y;
    let mut d = (dr * dr + dy * dy).sqrt();
    let max_reach = 0.99 * (a + b);
    if d < 0.3 {
        let s = 0.3 / d.max(1.0e-6);
        dr *= s;
        dy *= s;
        d = 0.3;
    }
    if d > max_reach {
        let s = max_reach / d;
        dr *= s;
        dy *= s;
        d = max_reach;
    }

    // Angles measured from vertical (+y), positive toward r.
    let beta = dy.atan2(dr); // angle of the target from horizontal-r… converted below
    let beta = core::f32::consts::FRAC_PI_2 - beta; // …to from-vertical
    let gamma = ((a * a + d * d - b * b) / (2.0 * a * d))
        .clamp(-1.0, 1.0)
        .acos();
    let th1 = beta - gamma; // elbow-up

    let dir = |ang: f32| r * ang.sin() + up * ang.cos();
    let elbow = SHOULDER + dir(th1) * a;

    // Forearm: straight from the elbow to the (clamped) target.
    let planar_target = SHOULDER + r * dr + up * dy;
    let fore = (planar_target - elbow).normalize();
    let wrist = elbow + fore * ARM_L2;
    let hand_end = wrist + fore * ARM_L3;

    let lat = r.cross(up).normalize();
    let spread = finger_spread(grip);
    let f1 = hand_end + fore * ARM_L4 + lat * spread;
    let f2 = hand_end + fore * ARM_L4 - lat * spread;
    Joints {
        elbow,
        wrist,
        hand_end,
        f1,
        f2,
        tip: fore,
    }
}

// ── The choreography (HTML phase labels, original-demo objective) ──────────
// HOME → APPROACH → LOWER → GRASP → LIFT → TRANSPORT → PLACE → RELEASE →
// RETREAT → GREET (wave) → BUMP (undo the fix) → HOME. Loops seamlessly.

/// Fingers extend `ARM_L4` past the hand; the carried H hangs at their tips.
const CARRY_OFS: f32 = ARM_L4 + 0.01;

struct Pose {
    hand: Vec3,
    grip: f32,
}

fn pose_at(t: f32) -> Pose {
    let home = Vec3::new(0.75, 1.55, 0.35);
    let hover = DROP_POS + Vec3::new(0.0, 0.62, 0.0);
    let grasp = DROP_POS + Vec3::new(0.0, CARRY_OFS + 0.02, 0.0);
    let lift = DROP_POS + Vec3::new(0.0, 0.75, 0.0);
    let above_slot = SLOT_POS + Vec3::new(0.0, 0.62, 0.0);
    let place = SLOT_POS + Vec3::new(0.0, CARRY_OFS + 0.02, 0.0);
    let wave_base = Vec3::new(0.35, 1.95, 0.50);
    let bump_pt = SLOT_POS + Vec3::new(0.14, 0.14, 0.05);

    let (hand, grip) = if t < 1.0 {
        (home, 0.2) // HOME
    } else if t < 2.6 {
        (lerp3(home, hover, smooth((t - 1.0) / 1.6)), 0.2) // APPROACHING
    } else if t < 3.2 {
        (lerp3(hover, grasp, smooth((t - 2.6) / 0.6)), 0.2) // LOWERING
    } else if t < T_GRAB {
        (grasp, 0.2 + 0.65 * smooth((t - 3.2) / 0.4)) // GRASPING
    } else if t < 4.8 {
        (lerp3(grasp, lift, smooth((t - T_GRAB) / 1.2)), 0.85) // LIFTING
    } else if t < 6.6 {
        (lerp3(lift, above_slot, smooth((t - 4.8) / 1.8)), 0.85) // TRANSPORTING
    } else if t < T_SNAP_START {
        (lerp3(above_slot, place, smooth((t - 6.6) / 0.6)), 0.85) // PLACING
    } else if t < T_RELEASE {
        (place, 0.85 - 0.65 * smooth((t - T_SNAP_START) / 0.7)) // RELEASING
    } else if t < 9.2 {
        (
            lerp3(
                place,
                place + Vec3::new(0.0, 0.45, 0.0),
                smooth((t - T_RELEASE) / 1.3),
            ),
            0.2,
        ) // RETREATING
    } else if t < 10.4 {
        (
            lerp3(
                place + Vec3::new(0.0, 0.45, 0.0),
                wave_base,
                smooth((t - 9.2) / 1.2),
            ),
            0.5,
        ) // GREETING (rise)
    } else if t < 13.5 {
        // GREETING: wave to the camera.
        let env = smooth((t - 10.7) / 0.4) * (1.0 - smooth((t - 13.1) / 0.4));
        let ph = 2.0 * core::f32::consts::PI * 1.7 * (t - 10.7);
        (wave_base + Vec3::new(0.38 * env * ph.sin(), 0.0, 0.0), 0.5) // GREETING
    } else if t < 14.2 {
        // BUMPING: the retract sweep deliberately clips the H's top corner.
        let u = smooth((t - 13.5) / 0.7);
        (lerp3(wave_base, bump_pt, u), 0.2)
    } else {
        (lerp3(bump_pt, home, smooth((t - 14.2) / 1.2)), 0.2) // RETURNING → HOME
    };
    Pose { hand, grip }
}

// ── Letter-H choreography ───────────────────────────────────────────────────

struct LetterState {
    pos: Vec3,
    yaw: f32,
    carrying: f32,
}

/// The H tumbles off the table back to its floor drop pose. `u` ∈ 0..1.
fn fall(u: f32) -> (Vec3, f32) {
    let u = u.clamp(0.0, 1.0);
    let ease = u * u; // accelerating — reads as gravity
    let mut p = SLOT_POS + (DROP_POS - SLOT_POS) * u;
    p.y = SLOT_POS.y
        + (DROP_POS.y - SLOT_POS.y) * ease
        + 0.15 * (u * core::f32::consts::PI).sin() * (1.0 - u);
    (p, DROP_YAW * smooth(ease))
}

fn letter_at(t: f32, joints: &Joints) -> LetterState {
    if t < T_GRAB {
        return LetterState {
            pos: DROP_POS,
            yaw: DROP_YAW,
            carrying: 0.0,
        };
    }
    if t < T_RELEASE {
        // Carried: hang at the finger tips, un-yaw in flight, snap into the slot.
        let tracked = joints.hand_end + joints.tip * CARRY_OFS;
        let yaw = DROP_YAW * (1.0 - smooth((t - T_GRAB) / (T_YAW_END - T_GRAB)));
        let pos = if t >= T_SNAP_START {
            let b = smooth((t - T_SNAP_START) / (T_RELEASE - T_SNAP_START));
            tracked + (SLOT_POS - tracked) * b
        } else {
            tracked
        };
        return LetterState {
            pos,
            yaw,
            carrying: 1.0,
        };
    }
    if t < T_BUMP_HIT {
        return LetterState {
            pos: SLOT_POS,
            yaw: 0.0,
            carrying: 0.0,
        };
    }
    let (pos, yaw) = fall((t - T_BUMP_HIT) / FALL_DUR);
    LetterState {
        pos,
        yaw,
        carrying: 0.0,
    }
}

// ── Rendering plumbing ──────────────────────────────────────────────────────

struct Frame {
    joints: Joints,
    letter: LetterState,
}

fn frame_at(t: f32) -> Frame {
    let pose = pose_at(t);
    let joints = ik(pose.hand, pose.grip);
    let letter = letter_at(t, &joints);
    Frame { joints, letter }
}

fn render(f: &Frame, w: u32, h: u32, pixels: &mut Array<Vec3>) -> Result<Vec<Vec3>> {
    let n = (w * h) as usize;
    launch!(
        draw_arm7,
        dim = n,
        (
            f.joints.elbow,
            f.joints.wrist,
            f.joints.hand_end,
            f.joints.f1,
            f.joints.f2,
            f.letter.pos,
            f.letter.yaw,
            f.letter.carrying,
            CAM_AZ,
            CAM_EL,
            CAM_DIST,
            w,
            h,
            &mut *pixels
        )
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
    let mut a = Args {
        width: 960,
        height: 540,
        frames: None,
        still: None,
    };
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
                a.still = Some((
                    argv[i + 1].parse().expect("--still <t> <name>"),
                    argv[i + 2].clone(),
                ));
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
    let n = (args.width * args.height) as usize;
    let mut pixels = Array::<Vec3>::zeros(n)?;

    // ---- still frame ----
    if let Some((t, name)) = &args.still {
        let f = frame_at(t % LOOP_T);
        let buf = render(&f, args.width, args.height, &mut pixels)?;
        image::write_ppm(&format!("{name}.ppm"), &buf, args.width, args.height)?;
        image::write_png(&format!("{name}.png"), &buf, args.width, args.height)?;
        println!("✓ wrote {name}.ppm / {name}.png (t = {t})");
        return Ok(());
    }

    // ---- CPU/GPU parity ----
    if std::env::args().any(|a| a == "--parity") {
        const PW: u32 = 128;
        const PH: u32 = 72;
        let pn = (PW * PH) as usize;
        let mut small = Array::<Vec3>::zeros(pn)?;
        let f = frame_at(5.5); // mid-carry: arm, held H, letters all in frame
        launch!(
            draw_arm7,
            dim = pn,
            (
                f.joints.elbow,
                f.joints.wrist,
                f.joints.hand_end,
                f.joints.f1,
                f.joints.f2,
                f.letter.pos,
                f.letter.yaw,
                f.letter.carrying,
                CAM_AZ,
                CAM_EL,
                CAM_DIST,
                PW,
                PH,
                &mut small
            )
        )?;
        let a = small.to_vec()?;
        let mut b = vec![Vec3::ZERO; pn];
        shannon_cpu::draw_arm7(
            f.joints.elbow,
            f.joints.wrist,
            f.joints.hand_end,
            f.joints.f1,
            f.joints.f2,
            f.letter.pos,
            f.letter.yaw,
            f.letter.carrying,
            CAM_AZ,
            CAM_EL,
            CAM_DIST,
            PW,
            PH,
            &mut b,
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
        // Robust criterion: branch-boundary pixels may legitimately differ.
        assert!(
            over as f32 / pn as f32 <= 0.005,
            "{over}/{pn} pixels exceed 1e-3"
        );
        assert!(worst <= 5e-2, "worst channel delta {worst}");
        println!(
            "✓ CPU == GPU at {PW}×{PH} ({:.2}% >1e-3, worst {worst:.2e})",
            100.0 * over as f32 / pn as f32
        );
        return Ok(());
    }

    // ---- headless benchmark ----
    if let Some(count) = args.frames {
        // Warm-up: first launch pays the one-time PTX module load.
        let _ = render(&frame_at(0.0), args.width, args.height, &mut pixels)?;
        let start = Instant::now();
        for k in 0..count {
            let t = (k as f32 / 30.0) % LOOP_T; // simulate 30 fps timeline
            let f = frame_at(t);
            let _ = render(&f, args.width, args.height, &mut pixels)?;
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
        "shannon-autograd — ARM-7 pick & place (Esc quits)",
        args.width as usize,
        args.height as usize,
        WindowOptions::default(),
    )?;
    window.set_target_fps(60);
    eprintln!("[arm_pick_place] window opened — rendering. Esc to quit.");

    let mut u32buf = vec![0u32; n];
    let start = Instant::now();
    let mut frames = 0u32;
    let mut last_report = Instant::now();

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let t = start.elapsed().as_secs_f32() % LOOP_T;
        let f = frame_at(t);
        let buf = render(&f, args.width, args.height, &mut pixels)?;

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
            println!(
                "{:.1} fps",
                frames as f32 / last_report.elapsed().as_secs_f32()
            );
            frames = 0;
            last_report = Instant::now();
        }
    }
    Ok(())
}

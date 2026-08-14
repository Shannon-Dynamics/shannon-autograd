//! Task 1.6 verify: CPU adapters vs the analytic result.

#[test]
fn forward_matches_analytic() {
    let a: Vec<f32> = (0..257).map(|i| i as f32 * 0.25 - 8.0).collect();
    let mut y = vec![0.0f32; a.len()];
    shannon_cpu::affine(&a, 2.0, 1.0, &mut y);
    for i in 0..a.len() {
        assert_eq!(y[i], a[i] * 2.0 + 1.0, "at {i}");
    }
}

#[test]
fn adjoint_matches_analytic_and_accumulates() {
    let adj_y = vec![1.0f32; 100];
    let mut adj_a = vec![0.0f32; 100];
    shannon_cpu::adj_affine(&adj_y, 2.0, &mut adj_a);
    assert!(adj_a.iter().all(|&g| g == 2.0));

    // Adjoints ACCUMULATE — a second pass must double, not overwrite.
    shannon_cpu::adj_affine(&adj_y, 2.0, &mut adj_a);
    assert!(adj_a.iter().all(|&g| g == 4.0));
}

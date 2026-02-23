import torch
import time
from PScan import pscan


def sequential_pscan_diag(A, X):
    B, L, C, D, D2 = X.shape
    A_mat = torch.zeros(B, L, C, D, D, dtype=A.dtype, device=A.device)
    for i in range(D):
        A_mat[..., i, i] = A[..., i]

    Y_list = [X[:, 0]]
    for t in range(1, L):
        Y_t = torch.einsum("bcij,bcjk->bcik", A_mat[:, t], Y_list[t - 1]) + X[:, t]
        Y_list.append(Y_t)
    return torch.stack(Y_list, dim=1)


def sequential_pscan_mat(A, X):
    B, L, C, D, D2 = X.shape
    if A.ndim == 4:
        A_mat = torch.zeros(B, L, C, D, D, dtype=A.dtype, device=A.device)
        for i in range(D):
            A_mat[..., i, i] = A[..., i]
        A = A_mat

    Y_list = [X[:, 0]]
    for t in range(1, L):
        Y_t = torch.einsum("bcij,bcjk->bcik", A[:, t], Y_list[t - 1]) + X[:, t]
        Y_list.append(Y_t)
    return torch.stack(Y_list, dim=1)


def check_forward_diag():
    print("=" * 60)
    print("测试前向传播（对角模式）")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 16, 4, 2

    A = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    Y_seq = sequential_pscan_diag(A, X)
    Y_par = pscan(A, X)

    max_diff = (Y_seq - Y_par).abs().max().item()
    mean_diff = (Y_seq - Y_par).abs().mean().item()

    print(f"A 形状: {A.shape}")
    print(f"X 形状: {X.shape}")
    print(f"Y 形状: {Y_par.shape}")
    print(f"最大差异: {max_diff:.2e}")
    print(f"平均差异: {mean_diff:.2e}")

    passed = max_diff < 1e-4
    print(f"测试通过: {passed}")
    print()
    return passed


def check_forward_mat():
    print("=" * 60)
    print("测试前向传播（矩阵模式）")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 16, 4, 2

    A = torch.randn(B, L, C, D, D, dtype=torch.complex64, device="cuda") * 0.3
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    Y_seq = sequential_pscan_mat(A, X)
    Y_par = pscan(A, X)

    max_diff = (Y_seq - Y_par).abs().max().item()
    mean_diff = (Y_seq - Y_par).abs().mean().item()

    print(f"A 形状: {A.shape}")
    print(f"X 形状: {X.shape}")
    print(f"Y 形状: {Y_par.shape}")
    print(f"最大差异: {max_diff:.2e}")
    print(f"平均差异: {mean_diff:.2e}")

    passed = max_diff < 1e-4
    print(f"测试通过: {passed}")
    print()
    return passed


def check_backward_diag():
    print("=" * 60)
    print("测试反向传播（对角模式）")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 16, 4, 2

    A_data = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
    X_data = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    A_seq = A_data.clone().requires_grad_(True)
    X_seq = X_data.clone().requires_grad_(True)
    A_par = A_data.clone().requires_grad_(True)
    X_par = X_data.clone().requires_grad_(True)

    Y_seq = sequential_pscan_diag(A_seq, X_seq)
    loss_seq = Y_seq.abs().pow(2).sum()
    loss_seq.backward()

    Y_par = pscan(A_par, X_par)
    loss_par = Y_par.abs().pow(2).sum()
    loss_par.backward()

    dA_max_diff = (A_seq.grad - A_par.grad).abs().max().item()
    dA_mean_diff = (A_seq.grad - A_par.grad).abs().mean().item()
    dX_max_diff = (X_seq.grad - X_par.grad).abs().max().item()
    dX_mean_diff = (X_seq.grad - X_par.grad).abs().mean().item()

    print(f"dA 最大差异: {dA_max_diff:.2e}")
    print(f"dA 平均差异: {dA_mean_diff:.2e}")
    print(f"dX 最大差异: {dX_max_diff:.2e}")
    print(f"dX 平均差异: {dX_mean_diff:.2e}")

    passed = dA_max_diff < 1e-3 and dX_max_diff < 1e-3
    print(f"测试通过: {passed}")
    print()
    return passed


def check_backward_mat():
    print("=" * 60)
    print("测试反向传播（矩阵模式）")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 16, 4, 2

    A_data = torch.randn(B, L, C, D, D, dtype=torch.complex64, device="cuda") * 0.3
    X_data = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    A_seq = A_data.clone().requires_grad_(True)
    X_seq = X_data.clone().requires_grad_(True)
    A_par = A_data.clone().requires_grad_(True)
    X_par = X_data.clone().requires_grad_(True)

    Y_seq = sequential_pscan_mat(A_seq, X_seq)
    loss_seq = Y_seq.abs().pow(2).sum()
    loss_seq.backward()

    Y_par = pscan(A_par, X_par)
    loss_par = Y_par.abs().pow(2).sum()
    loss_par.backward()

    dA_max_diff = (A_seq.grad - A_par.grad).abs().max().item()
    dA_mean_diff = (A_seq.grad - A_par.grad).abs().mean().item()
    dX_max_diff = (X_seq.grad - X_par.grad).abs().max().item()
    dX_mean_diff = (X_seq.grad - X_par.grad).abs().mean().item()

    print(f"dA 最大差异: {dA_max_diff:.2e}")
    print(f"dA 平均差异: {dA_mean_diff:.2e}")
    print(f"dX 最大差异: {dX_max_diff:.2e}")
    print(f"dX 平均差异: {dX_mean_diff:.2e}")

    passed = dA_max_diff < 1e-3 and dX_max_diff < 1e-3
    print(f"测试通过: {passed}")
    print()
    return passed


def check_various_shapes():
    print("=" * 60)
    print("测试多种形状")
    print("=" * 60)

    shapes = [
        (1, 8, 1, 2),
        (2, 16, 4, 2),
        (4, 32, 8, 2),
        (1, 64, 2, 2),
        (2, 128, 4, 2),
    ]

    all_passed = True
    for B, L, C, D in shapes:
        torch.manual_seed(42)

        A_diag = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
        A_mat = torch.randn(B, L, C, D, D, dtype=torch.complex64, device="cuda") * 0.3
        X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

        Y_seq_diag = sequential_pscan_diag(A_diag, X)
        Y_par_diag = pscan(A_diag, X)
        diff_diag = (Y_seq_diag - Y_par_diag).abs().max().item()

        Y_seq_mat = sequential_pscan_mat(A_mat, X)
        Y_par_mat = pscan(A_mat, X)
        diff_mat = (Y_seq_mat - Y_par_mat).abs().max().item()

        passed_diag = diff_diag < 1e-4
        passed_mat = diff_mat < 1e-4
        passed = passed_diag and passed_mat

        print(f"形状 (B={B}, L={L}, C={C}, D={D}): 对角={diff_diag:.2e} 矩阵={diff_mat:.2e} 通过={passed}")
        all_passed = all_passed and passed

    print()
    return all_passed


def check_long_sequence():
    print("=" * 60)
    print("测试长序列")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 512, 4, 2

    A = torch.randn(B, L, C, D, D, dtype=torch.complex64, device="cuda") * 0.2
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    Y_seq = sequential_pscan_mat(A, X)
    Y_par = pscan(A, X)

    max_diff = (Y_seq - Y_par).abs().max().item()
    mean_diff = (Y_seq - Y_par).abs().mean().item()

    print(f"A 形状: {A.shape}")
    print(f"X 形状: {X.shape}")
    print(f"最大差异: {max_diff:.2e}")
    print(f"平均差异: {mean_diff:.2e}")

    passed = max_diff < 1e-3
    print(f"测试通过: {passed}")
    print()
    return passed


def check_4d_input():
    print("=" * 60)
    print("测试 4D 输入（自动压缩）")
    print("=" * 60)
    torch.manual_seed(42)
    B, L, C, D = 2, 16, 4, 2

    A = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda") * 0.5
    X = torch.randn(B, L, C, D, dtype=torch.complex64, device="cuda")

    Y_par = pscan(A, X)

    print(f"A 形状: {A.shape}")
    print(f"X 形状: {X.shape}")
    print(f"Y 形状: {Y_par.shape}")

    passed = Y_par.shape == X.shape
    print(f"输出形状与输入一致: {passed}")
    print()
    return passed


def benchmark():
    print("=" * 60)
    print("性能基准")
    print("=" * 60)

    torch.manual_seed(42)
    B, L, C, D = 8, 256, 16, 2

    A = torch.randn(B, L, C, D, D, dtype=torch.complex64, device="cuda") * 0.3
    X = torch.randn(B, L, C, D, 1, dtype=torch.complex64, device="cuda")

    for _ in range(3):
        _ = pscan(A, X)

    torch.cuda.synchronize()
    n_iters = 100

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = sequential_pscan_mat(A, X)
    torch.cuda.synchronize()
    seq_time = (time.time() - start) / n_iters * 1000

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = pscan(A, X)
    torch.cuda.synchronize()
    par_time = (time.time() - start) / n_iters * 1000

    print(f"形状: B={B}, L={L}, C={C}, D={D}")
    print(f"串行耗时: {seq_time:.3f} ms")
    print(f"并行耗时: {par_time:.3f} ms")
    print(f"加速比: {seq_time / par_time:.2f}x")
    print()


def main():
    print("=" * 60)
    print("PScan 正确性检查")
    print("=" * 60)
    print()

    results = {}

    results["前向_对角"] = check_forward_diag()
    results["前向_矩阵"] = check_forward_mat()
    results["反向_对角"] = check_backward_diag()
    results["反向_矩阵"] = check_backward_mat()
    results["多形状"] = check_various_shapes()
    results["长序列"] = check_long_sequence()
    results["4d输入"] = check_4d_input()

    print("=" * 60)
    print("汇总")
    print("=" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "通过" if passed else "失败"
        print(f"{name}: {status}")
        all_passed = all_passed and passed

    print()
    print(f"总体结果: {'全部测试通过' if all_passed else '部分测试失败'}")
    print()

    benchmark()

    return all_passed


if __name__ == "__main__":
    main()
    

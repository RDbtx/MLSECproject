from autoattack import AutoAttack
from src.setup_utils import *
from src.results_processing_utils import *
import time
import torch


def robust_acc_autoattack(model, x, y, eps: float, device: str, mode: str = "standard", bs: int = 50,
                          verbose: bool = True) -> float:
    if mode == "fast":
        print("Fast mode selected [APGD-CE only]!")
        adversary = AutoAttack(model, norm="Linf", eps=eps, version="custom", device=device, verbose=verbose)
        adversary.attacks_to_run = ["apgd-ce"]

    elif mode == "untargeted":
        print("Untargeted mode selected [APGD-CE, SQUARE]!")
        adversary = AutoAttack(model, norm="Linf", eps=eps, version="custom", device=device, verbose=verbose)
        adversary.attacks_to_run = ["apgd-ce", "square"]

    elif mode == "targeted":
        print("Targeted mode selected [APGD-T, FAB-T]!")
        adversary = AutoAttack(model, norm="Linf", eps=eps, version="custom", device=device, verbose=verbose)
        adversary.attacks_to_run = ["apgd-t", "fab-t"]

    elif mode == "standard":
        print("Running standard mode!")
        adversary = AutoAttack(model, norm="Linf", eps=eps, version="standard", device=device, verbose=verbose)

    else:
        raise ValueError("Unknown mode. Use one of: fast, untargeted, targeted, standard")

    x_adv = adversary.run_standard_evaluation(x, y, bs=bs)

    with torch.no_grad():
        pred = model(x_adv).argmax(1)
        return (pred == y).float().mean().item()


def autoattack_models(models: dict, x_test_cpu, y_test_cpu, batch_size: int, mode: str = "standard",
                      verbose: bool = True) -> dict:
    print("\n---- STARTING AUTOATTACK ----")
    results = {}

    steps = [1, 4, 8, 12, 16]
    eps_list = [e / 255 for e in steps]

    for step, eps in zip(steps, eps_list):
        print(f"\nCURRENT EPSILON: {step}/255")
        eps_key = f"{step}/255"
        results[eps_key] = {}

        # cache x/y per device for this epsilon (avoids repeated transfers)
        x_cache = {}
        y_cache = {}

        for name, model in models.items():
            dev = device_for_model(name)

            # move model to correct device
            model = model.to(dev).eval()
            move_model_extras_to_device(model, dev)

            # move data once per device
            if dev not in x_cache:
                x_cache[dev] = x_test_cpu.to(dev)
                y_cache[dev] = y_test_cpu.to(dev)
            x = x_cache[dev]
            y = y_cache[dev]

            print(f"\ntesting {name} on {dev}")
            start = time.perf_counter()

            acc = robust_acc_autoattack(
                model, x, y, eps,
                device=dev,
                mode=mode,
                bs=batch_size,
                verbose=verbose
            )

            end = time.perf_counter()
            h, m, s = compute_elapsed_time(start, end)

            results[eps_key][name] = {
                "robust_acc": acc,
                "device": dev,
                "time_s": float(end - start),
                "mode": mode,
                "bs": batch_size
            }

            print(f"eps={step:>2d}/255  {name}: {acc:.3f}")
            print(f"elapsed time for {name}: {h:02d}:{m:02d}:{s:02d}")

    return results


def compute_autoattacks(models: list, samples: int = 200, seeds: int = 0, batch_size: int = 50, mode: str = "standard",
                        verbose: bool = True) -> dict:
    models = load_models(models)
    x_test, y_test = load_data(dataset_samples=samples, seed=seeds, batch_size=batch_size)
    results = autoattack_models(
        models,
        x_test_cpu=x_test,
        y_test_cpu=y_test,
        batch_size=batch_size,
        mode=mode,
        verbose=verbose
    )

    save_results(results)
    return results

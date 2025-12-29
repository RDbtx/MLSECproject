from autoattack import AutoAttack
from src.setup_utils import *
from src.results_processing_utils import *
import time
import torch


@torch.no_grad()
def clean_acc(model, x, y):
    pred = model(x).argmax(1)
    return (pred == y).float().mean().item()


def robust_acc_autoattack(model, x, y, eps, device, bs=50):
    adversary = AutoAttack(
        model, norm="Linf", eps=eps, version="standard", device=device, verbose=True
    )
    x_adv = adversary.run_standard_evaluation(x, y, bs=bs)
    with torch.no_grad():
        pred = model(x_adv).argmax(1)
        return (pred == y).float().mean().item()


def autoattack_models(models, x_test_cpu, y_test_cpu):
    print("\n---- STARTING AUTOATTACK ----")
    results = {}

    steps = [1, 2, 4, 8, 12, 16]
    eps_list = [e / 255 for e in steps]

    for step, eps in zip(steps, eps_list):
        print(f"\nCURRENT EPSILON: {step}/255")
        eps_key = f"{step}/255"
        results[eps_key] = {}

        for name, model in models.items():
            dev = device_for_model(name)

            model = model.to(dev).eval()
            move_model_extras_to_device(model, dev)
            x = x_test_cpu.to(dev)
            y = y_test_cpu.to(dev)

            print(f"\ntesting {name} on {dev}")
            start = time.perf_counter()

            acc = robust_acc_autoattack(model, x, y, eps, device=dev, bs=50)

            end = time.perf_counter()
            h, m, s = compute_elapsed_time(start, end)

            results[eps_key][name] = {"robust_acc": acc, "device": dev, "time_s": float(end - start)}

            print(f"eps={step:>2d}/255  {name}: {acc:.3f}")
            print(f"elapsed time for {name}: {h:02d}:{m:02d}:{s:05.2f}")

    return results

def compute_autoattacks(models : list, samples: int = 200, seeds: int = 0,  batch_size: int = 50):
    models = load_models(models)
    x_test, y_test = load_data(dataset_samples=samples, seed=seeds, batch_size=batch_size)

    results = autoattack_models(models, x_test, y_test)
    save_results(results)
    return results

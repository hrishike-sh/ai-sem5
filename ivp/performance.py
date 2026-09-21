"""CPU budgets for the single active video worker; no model or device changes."""
import os


def available_cpu_threads():
    """Use available logical processors, respecting affinity where supported."""
    affinity = getattr(os, 'sched_getaffinity', None)
    if affinity is not None:
        try:
            count = len(affinity(0))
            if count:
                return count
        except (OSError, NotImplementedError):
            pass
    return max(1, os.cpu_count() or 1)


def configure_opencv_threads():
    import cv2
    count = available_cpu_threads()
    cv2.setNumThreads(count)
    return {'cpu_policy': 'all available logical processors',
            'available_logical_processors': count,
            'opencv_threads': cv2.getNumThreads()}


def configure_torch_threads(torch):
    torch.set_num_threads(available_cpu_threads())
    return torch.get_num_threads()

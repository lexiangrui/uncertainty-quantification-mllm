import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visual_embedding_sweep import summarize_trials, validate_sigmas


def _trial(sigma, seed, changed, delta_nll, kl, relative_l2):
    return {
        "sigma": sigma,
        "seed": seed,
        "answer_changed": changed,
        "delta_nll": delta_nll,
        "kl": kl,
        "geometry": {
            "mean_token_relative_l2": relative_l2,
            "mean_token_cosine": 1.0 - relative_l2 / 2.0,
        },
    }


def _assert_raises(function, *args):
    try:
        function(*args)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_validate_sigmas_requires_strictly_increasing_positive_grid():
    assert validate_sigmas([0.1, 0.5, 1.0]) == (0.1, 0.5, 1.0)
    _assert_raises(validate_sigmas, [0.1, 0.1])
    _assert_raises(validate_sigmas, [0.0, 0.1])


def test_summary_reports_curve_auc_and_censored_first_flip():
    sigmas = (0.5, 1.0)
    seeds = (42, 43)
    trials = [
        _trial(0.5, 42, False, 0.1, 0.01, 0.4),
        _trial(0.5, 43, False, -0.1, 0.03, 0.4),
        _trial(1.0, 42, True, 0.3, 0.05, 0.8),
        _trial(1.0, 43, False, -0.2, 0.07, 0.8),
    ]
    summary = summarize_trials(trials, sigmas, seeds)
    assert summary["curve"][0]["flip_rate"] == 0.0
    assert summary["curve"][1]["flip_rate"] == 0.5
    assert summary["first_flip_rate"] == 0.5
    assert summary["median_first_flip_sigma"] == 1.0
    assert summary["no_flip_through_max_sigma_seeds"] == [43]
    assert abs(summary["flip_auc"] - 0.125) < 1e-12
    assert abs(summary["abs_delta_nll_auc"] - 0.1125) < 1e-12


def test_summary_rejects_missing_grid_cell():
    _assert_raises(
        summarize_trials,
        [_trial(0.5, 42, False, 0.0, 0.0, 0.4)],
        (0.5,),
        (42, 43),
    )


if __name__ == "__main__":
    test_validate_sigmas_requires_strictly_increasing_positive_grid()
    test_summary_reports_curve_auc_and_censored_first_flip()
    test_summary_rejects_missing_grid_cell()
    print("visual embedding sweep tests passed")

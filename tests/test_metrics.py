import numpy as np
import pytest

from mulg.metrics import class7, evaluate


def test_binary_neutral_conventions_and_continuous_labels():
    result = evaluate([-0.1, 0, 0.1], [-0.1, -0.1, -0.1])
    assert result["Acc2"] == 50
    assert result["Acc2_with_zero"] == pytest.approx(100 / 3)
    assert result["n_binary_nonzero"] == 2
    assert result["Acc7"] == 100


def test_seven_classes_clipping_ties_and_classifier_argmax():
    np.testing.assert_array_equal(class7([-4, -2.5, -1.5, 0.5, 1.5, 2.5, 4]), [0, 1, 1, 3, 5, 5, 6])
    # A classifier's maximum-probability class can differ from its expected score.
    result = evaluate([3], [0.2], classes=[6])
    assert result["Acc7"] == 100
    assert result["Acc2"] == 100
    assert evaluate([0], [0])["Acc2"] is None
    with pytest.raises(ValueError):
        evaluate([1], [float("nan")])
